from BPEtokenizer import BPEtokenizer
import os
from typing import BinaryIO
import tqdm
import time
import numpy as np

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

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description="Tokenize the file",
    )

    parser.add_argument("--input", type=str ,default="./end_poem.txt", help="Input path of dataset to be tokenized.")
    parser.add_argument("--merges", type=str, default="./TinyStories_merges.pkl", help="input merges file")
    parser.add_argument("--vocab", type=str, default="./TinyStories_vocab.pkl", help="input vocab file")
    parser.add_argument("--special_tokens", type=str ,default="./special_tokens.txt", help="Input path of special tokens. Split by \\n.")

    parser.add_argument("--output", type=str ,default="./end_poem.bin", help="Output path of the tokenizer")

    # parser.add_argument("--itr_mode", type=bool, default=False, help="Flag for iterable tokenize", action='store_true')
    parser.add_argument("--num_chunks", type=int, default=1, help="Approx # of chunks")

    args = parser.parse_args()


    tokenizer = BPEtokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])

    num_tokens = 0
    num_bytes = 0

    with open(args.input, "rb") as f:
        boundaries = find_chunk_boundaries(f, args.num_chunks, tokenizer.endoftext_token)
        # handle each chunk
        with open(args.output, "wb") as o:
            # process, and CAN BE concurrent! AND THIS FEATURES LATER USING pool.imap etc.
            start_time = time.time()
            pbar = tqdm.tqdm(total=len(boundaries)-1, desc="Encoding chunks", initial=0)
            for start, end in zip(boundaries[:-1],boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="replace")

                encoded_chunk = tokenizer.encode(chunk)

                num_tokens = num_tokens + len(encoded_chunk)
                num_bytes = num_bytes + (end - start)

                chunk = np.array(encoded_chunk, dtype=np.uint16)
                o.write(chunk.tobytes())
                pbar.update(1)
            end_time = time.time()
            pbar.close()

    print(f"Compression ratio(bytes/tokens):{num_bytes} / {num_tokens} = {num_bytes / num_tokens}")
    print(f"processing rate(tokens/sec):{num_tokens} / {(end_time - start_time)} = { num_tokens / (end_time - start_time)}")
    