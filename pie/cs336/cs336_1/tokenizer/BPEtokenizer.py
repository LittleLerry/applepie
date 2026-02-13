from typing import Iterable
from copy import deepcopy
import regex as re
# example usage:
# tokenizer: BPEtokenizer = ...
# with open(corpus_path, encoding='utf-8') as f:
#   tokenizer.encode(f.read()) 
class BPEtokenizer():

    def __init__(
        self,
        vocab:dict[int,bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.rvocab: dict[bytes, int] = {}
        for k, v in self.vocab.items():
            self.rvocab[v] = k
 
    @classmethod
    def from_files(cls, vocab_filepath,merges_filepath,special_tokens=None):
        import pickle
        assert vocab_filepath.endswith('.pkl') and merges_filepath.endswith('.pkl')
        
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        
        if isinstance(special_tokens, str): # means it's a path
            _tokens = []
            with open(merges_filepath, mode="r") as f:
                 tks = f.readlines()
                 for tk in tks:
                    _tokens.append(tk.strip())
            return cls(vocab,merges,_tokens)
        else:
            return cls(vocab,merges,special_tokens)

    # return list[int]
    def encode(self, text: str):
        # empty strings
        if (len(text) == 0):
            return []

        chunks_index: list[tuple[int, int]] = []
        result: list[int] = []
        special = set()

        # pre-pre tokenization
        if self.special_tokens is not None:
            escaped_tokens = sorted((re.escape(token) for token in self.special_tokens),key=len,reverse=True)
            pattern = re.compile('|'.join(escaped_tokens))

            last_end = 0
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if start > last_end:
                    chunks_index.append((last_end,start))
                
                chunks_index.append((start, end))
                special.add((start, end))

                last_end = end

            if last_end < len(text):
                chunks_index.append((last_end, len(text)))
        else:
            chunks_index.append((0, len(text)))

        for start, end in chunks_index:
            if (start, end) in special:
                result.append(self.rvocab[text[start:end].encode('utf-8')])
                continue
            # pre-tokenization
            tokens = re.finditer(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", text[start:end])
            for token in tokens:
                # tokenization
                bytes_to_merged = list(bytes([b]) for b in token.group().encode('utf-8', errors='ignore'))
                temp_result:list[bytes] = []

                assert(len(bytes_to_merged) > 0)
                while(True):
                    merged = False
                    if (len(bytes_to_merged) == 1):
                        result.append(self.rvocab[bytes_to_merged[0]])
                        break
                    for (left, right) in self.merges:
                        i = 0
                        while (i < len(bytes_to_merged) - 1):
                            if (bytes_to_merged[i] == left) and (bytes_to_merged[i+1] == right):
                                temp_result.append(bytes_to_merged[i]+bytes_to_merged[i+1])
                                i = i + 2
                                merged = True
                            else:
                                temp_result.append(bytes_to_merged[i])
                                i = i + 1
                        if(i == len(bytes_to_merged) - 1):
                            temp_result.append(bytes_to_merged[i])
                        # update
                        _temp_result = deepcopy(temp_result)
                        del bytes_to_merged
                        bytes_to_merged = _temp_result
                        temp_result.clear()
                
                    if not merged:
                        for b in bytes_to_merged:
                            result.append(self.rvocab[b])
                        break

        return result

    # return Iterator[int]
    def encode_iterable(self, iterable: Iterable[str]):
        pattern = None
        if self.special_tokens is not None:
            escaped_tokens = sorted((re.escape(token) for token in self.special_tokens),key=len,reverse=True)
            pattern = re.compile('|'.join(escaped_tokens))
        
        def get_parse_index(chunk: str) -> Iterable[tuple[bool, int, int]]:
            if pattern is None:
                yield (False, 0, len(chunk))
            else:
                last_end = 0
                for match in pattern.finditer(chunk):
                    start, end = match.start(), match.end()
                    if start > last_end:
                        yield (False, last_end, start)

                    yield (True, start, end)

                    last_end = end
                if last_end < len(chunk):
                    yield (False, last_end, len(chunk))

        for text in iterable:
            if (len(text) == 0):
                yield []
            else:
                for (is_special, start, end) in get_parse_index(text):
                    if (is_special):
                        yield self.rvocab[text[start:end].encode('utf-8')]
                    else:
                        tokens = re.finditer(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", text[start:end]) # tokens is iterable
                        for token in tokens:
                            # token is very short, directly perform matching
                            bytes_to_merged = list(bytes([b]) for b in token.group().encode('utf-8', errors='replace'))
                            temp_result:list[bytes] = []

                            assert(len(bytes_to_merged) > 0)
                            while(True):
                                merged = False
                                if (len(bytes_to_merged) == 1):
                                    yield self.rvocab[bytes_to_merged[0]]
                                    break
                                # 
                                for (left, right) in self.merges:
                                    i = 0
                                    while (i < len(bytes_to_merged) - 1):
                                        if (bytes_to_merged[i] == left) and (bytes_to_merged[i+1] == right):
                                            temp_result.append(bytes_to_merged[i]+bytes_to_merged[i+1])
                                            i = i + 2
                                            merged = True
                                        else:
                                            temp_result.append(bytes_to_merged[i])
                                            i = i + 1
                                    if(i == len(bytes_to_merged) - 1):
                                        temp_result.append(bytes_to_merged[i])
                                    # update
                                    _temp_result = deepcopy(temp_result)
                                    del bytes_to_merged
                                    bytes_to_merged = _temp_result
                                    temp_result.clear()
                
                                if not merged:
                                    for b in bytes_to_merged:
                                        yield self.rvocab[b]
                                    break
    # endo of the iterator

    # return str
    def decode(self, ids:list[int]):
        # decoder may not always success, since the combination of b''.join([self.vocab[k] for k in ids])
        # is not garanteed to be decode-able under utf-8
        return  (b''.join([self.vocab[k] for k in ids])).decode('utf-8', errors='replace')
    
    @property
    def endoftext_token(self):
        return self.vocab[0]
    
    @property
    def endoftext_token_id(self):
        return b'\x00'
