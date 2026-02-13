import os
from typing import BinaryIO, Iterator, Any

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# combine two dicts
def merge(
    source: dict[tuple[bytes, ...], int], 
    target: dict[tuple[bytes, ...], int],
) -> None:
    for k,v in target.items():
        source[k] = source.get(k, 0) + v

# workers' task
# special_tokens[0] must be <|endoftext|> token
def read_dataset(
    args,
) -> dict[tuple[bytes, ...], int]:
    filepath, special_tokens, start, end = args

    result = {}

    with open(filepath, 'rb') as f:
        import regex as re
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # docs = chunk.split(special_tokens[0])
        escaped_tokens = sorted((re.escape(token) for token in special_tokens),key=len,reverse=True)
        pattern = re.compile('|'.join(escaped_tokens))
        def docs_itr()-> Iterator[str]:
            last_end = 0
            for match in pattern.finditer(chunk):
                start, end = match.start(), match.end()
                if start > last_end:
                    yield chunk[last_end:start]
                # yield match.group(0)
                last_end = end
            if last_end < len(chunk):
                yield chunk[last_end:]

        for doc in docs_itr():
            tokens = re.finditer(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", doc)
            for token in tokens:
                merge(result, {tuple(bytes([b]) for b in token.group().encode("utf-8")): 1})
    
    return result
    
if __name__ == '__main__':

    from multiprocessing import Pool
    import argparse
    import tqdm
    import pickle
    import cProfile

    parser = argparse.ArgumentParser(
        description="Multiprocessing BPE Algorithm.",
        epilog="Example: python bpe.py"
    )

    parser.add_argument("--num_processes", type=int, default=8, help="# of workers in reading dataset.")
    parser.add_argument("--special_tokens", type=str ,default="./special_tokens.txt", help="Input path of special tokens. Split by \\n.")
    parser.add_argument("--train_dataset", type=str ,default="./tinystories_sample_5M.txt", help="Input path of train dataset.")
    parser.add_argument("--vocab_size", type=int, default=1000, help="vocab size")
    parser.add_argument("--output_merges", type=str, default="./merges.pkl", help="output merges file")
    parser.add_argument("--output_vocab", type=str, default="./vocab.pkl", help="output vocab file")

    args = parser.parse_args()

    profiler = cProfile.Profile()

    # get special tokens
    special_tokens = []
    with open(args.special_tokens, "r") as f:
        tks = f.readlines()
        for tk in tks:
            special_tokens.append(tk.strip())
    assert len(special_tokens) >= 1 # at least you need have special_tokens[0] = <|eos|> token
    
    # get chunk boundary
    chunks_index = []
    with open(args.train_dataset, "rb") as f:
        chunks_index = find_chunk_boundaries(f, args.num_processes, bytes(special_tokens[0], 'utf-8'))
    assert len(chunks_index) >= 2 and len(chunks_index) <= args.num_processes + 1

    # prepare tasks
    TASKS = [(args.train_dataset, special_tokens, start, end) for start,end in zip(chunks_index[:-1], chunks_index[1:])]

    # launch workers
    print("Launching workers to read and parse dataset")
    profiler.enable()
    with Pool(processes=len(TASKS)) as pool:
        # we will not use imap here cause the number of workers is not much less than the number of tasks
        results = pool.map(read_dataset, TASKS)

    # single process merging, handling the remaining part
    print("Merging results from works")
    records = {}
    for result in results:
        merge(records, result)
    profiler.disable()
    profiler.print_stats()

    def update(
        d: dict[bytes, dict[str, Any]],
        k: bytes,
        v: int,
        left: bytes,
        right: bytes,
    ) -> None:
        val = d.get(k, {"count":0})["count"] + v
        assert (val >= 0) and (k == left + right)
        d[k] = {"count":val, "left":left, "right": right}
        if d[k]["count"] == 0:
            d.pop(k)
    
    def generate_new_token(
        t: tuple[bytes, ...], 
        bytes_merge: bytes, 
        pair_freq: dict[bytes, dict[str, Any]], 
        occurences: int,
    ) -> Iterator[bytes]:
        i = 0
        token = list(t)
        while( i < len(token) - 1):
            if (token[i] + token[i+1] == bytes_merge):
                # removal i, i+1, add related freq
                if(i-1 >= 0):
                    update(pair_freq, token[i-1]+token[i], -occurences,token[i-1],token[i])
                    update(pair_freq, token[i-1]+bytes_merge, occurences,token[i-1],bytes_merge)
                if(i+2 <= len(token)-1):
                    update(pair_freq, token[i+1]+token[i+2], -occurences,token[i+1],token[i+2])
                    update(pair_freq, bytes_merge+token[i+2], occurences,bytes_merge,token[i+2])
                update(pair_freq, bytes_merge, -occurences,token[i],token[i+1])
                token[i+1] = bytes_merge # handle case where 'a x y b',xy==bytes_merge, but a is merged before

                yield bytes_merge
                i = i + 2
            else:
                yield token[i]
                i = i + 1

        if (i == len(token) - 1):
            yield token[i]
    
    def perform_merge(
        records: dict[tuple[bytes, ...], int], 
        pair_freq: dict[bytes, dict[str, Any]], 
        bytes_merge: bytes,
    ) -> bool:
        merged = False
        replace = {}

        for token, occurences in records.items():
            new_token = tuple(list(generate_new_token(token, bytes_merge, pair_freq, occurences))) # freq updated
            if (new_token != token):
                replace[token] = new_token
                merged = True
        # If not to do so: RuntimeError: dictionary keys changed during iteration
        for old_key, new_key in replace.items():
            if old_key in records:
                records[new_key] = records.pop(old_key)

        return merged
    
    pair_freq: dict[bytes, dict[str, Any]] = {}
    vocab: dict[int, bytes]= {}
    merges: list[tuple[bytes, bytes]] = []

    cur_size = 0
    for i in range(len(special_tokens)):
        vocab[i] = special_tokens[i].encode("utf-8")
        cur_size = cur_size + 1
    for i in range(256):
        vocab[cur_size] = bytes([i])
        cur_size = cur_size + 1
    for token, occurences in records.items():
        i = 0
        while (i < len(token) - 1):
            update(pair_freq, token[i] + token[i+1], occurences, token[i] , token[i+1])
            i = i + 1
        
    #profiler.enable()
    pbar = tqdm.tqdm(total=args.vocab_size, desc="Generating merges and vocab", initial=cur_size)
    while (cur_size < args.vocab_size):
        if(len(pair_freq) == 0):
            break

        max_key = max(pair_freq, key=lambda k: (pair_freq[k]['count'], pair_freq[k]['left'], pair_freq[k]['right']))
        # update first, cause perform merge will have side effects on those tokens
        merges.append(tuple([pair_freq[max_key]["left"], pair_freq[max_key]["right"]]))
        vocab[cur_size] = pair_freq[max_key]["left"] + pair_freq[max_key]["right"]

        perform_merge(records, pair_freq, max_key)

        cur_size = cur_size + 1
        pbar.update(1)
    pbar.close()
    #profiler.disable()
    #profiler.print_stats()
    
    print(f"Writing results to {args.output_merges} and {args.output_vocab}")
    with open(args.output_vocab, "wb") as f:
        pickle.dump(vocab, f)
    with open(args.output_merges, "wb") as f:
        pickle.dump(merges, f)

    print("Done")

    
    