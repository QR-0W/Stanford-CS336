"""
Script to encode full training datasets to numpy arrays.
"""

import time
import numpy as np
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def load_tokenizer(dataset_name):
    vocab_path = OUTPUT_DIR / f"{dataset_name}_vocab.json"
    merges_path = OUTPUT_DIR / f"{dataset_name}_merges.json"
    return Tokenizer.from_files(str(vocab_path), str(merges_path), ["<|endoftext|>"])


def encode_and_save(tokenizer, input_path, output_path, desc):
    print(f"Encoding {desc}...")
    start_time = time.time()

    def line_generator(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield line

    ids = []
    count = 0
    # Pre-allocate? Hard since we don't know exact token count.
    # List append is amortized O(1).
    # For 4 billion items, list might use a lot of pointers (8 bytes per pointer).
    # 4 billion * 8 bytes = 32GB RAM just for the list pointers + int objects.
    # We might run out of RAM (User has 256GB? No, usually typical VM).
    # Wait, check user info? "User has 1 active workspaces".
    # I don't know the RAM.
    # Use streaming to numpy maybe? Or chunks.

    # Safest approach: Write to disk in chunks? Or append to a list in chunks and verify memory?
    # Python 3.12 int is 28 bytes.
    # 4e9 * 28 bytes = 112 GB.
    # Plus list overhead.
    # It might be tight or OOM.

    # Better: Write to valid .npy file incrementally?
    # Numpy .save doesn't support append.
    # memmap?
    # We can guess file size.
    # Ratio ~ 4 bytes/token.
    # 12GB bytes -> 3B tokens.
    # 3B * 2 bytes = 6GB.
    # We can create a memmap of size 6GB (oversized?) and resize later?
    # Or just write raw bytes to a file and then load as numpy?
    # Yes, write raw uint16 binary data.

    raw_path = output_path.with_suffix(".bin")

    with open(raw_path, "wb") as f_out:
        with open(input_path, "r", encoding="utf-8") as f_in:
            chunk = []
            CHUNK_SIZE = 1_000_000

            for token_id in tokenizer.encode_iterable(f_in):
                chunk.append(token_id)
                if len(chunk) >= CHUNK_SIZE:
                    # Convert to bytes
                    arr = np.array(chunk, dtype=np.uint16)
                    f_out.write(arr.tobytes())
                    count += len(chunk)
                    chunk = []
                    print(f"  Processed {count / 1e6:.1f}M tokens...", end="\r")

            # Remaining
            if chunk:
                arr = np.array(chunk, dtype=np.uint16)
                f_out.write(arr.tobytes())
                count += len(chunk)

    print(f"\n  Total tokens: {count}")
    print(f"  Converting to .npy...")

    # Now load raw and save as npy (or just keep raw?)
    # Assignment recommends "serializing the token IDs as a NumPy array".
    # We can load the bin file as memmap and save to npy, or just use `fromfile`.

    data = np.fromfile(raw_path, dtype=np.uint16)
    np.save(output_path, data)

    # Cleanup raw
    raw_path.unlink()

    end_time = time.time()
    print(f"  Saved to {output_path}")
    print(f"  Time taken: {end_time - start_time:.2f}s")


def main():
    try:
        ts_tokenizer = load_tokenizer("tiny_stories")
        owt_tokenizer = load_tokenizer("owt")

        # TinyStories Train (2.1GB)
        # encode_and_save(ts_tokenizer, DATA_DIR / "TinyStoriesV2-GPT4-train.txt", OUTPUT_DIR / "tiny_stories_train.npy", "TinyStories Train")

        # OpenWebText Train (12GB)
        encode_and_save(owt_tokenizer, DATA_DIR / "owt_train.txt", OUTPUT_DIR / "owt_train.npy", "OpenWebText Train")

        # Also TinyStories if needed
        encode_and_save(
            ts_tokenizer,
            DATA_DIR / "TinyStoriesV2-GPT4-train.txt",
            OUTPUT_DIR / "tiny_stories_train.npy",
            "TinyStories Train",
        )

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
