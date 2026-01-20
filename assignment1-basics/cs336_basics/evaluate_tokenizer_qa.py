"""
Script to evaluate tokenizer performance and answer assignment questions (a)-(c).
"""

import time
import json
import numpy as np
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def load_tokenizer(dataset_name):
    vocab_path = OUTPUT_DIR / f"{dataset_name}_vocab.json"
    merges_path = OUTPUT_DIR / f"{dataset_name}_merges.json"

    # TinyStories might have different special tokens?
    # Based on training scripts:
    # TS: ["<|endoftext|>"]
    # OWT: ["<|endoftext|>"]
    # Be careful with the XML escaped version if copied from user request?
    # But files on disk should be correct if I used the scripts I pushed.
    # Actually, looking at my previous `cat` of `train_owt_bpe.py`, it had `SPECIAL_TOKENS = ["<|endoftext|>"]`
    # Let's assume standard "<|endoftext|>" for now.

    return Tokenizer.from_files(str(vocab_path), str(merges_path), ["<|endoftext|>"])


def read_samples(file_path, num_samples=10):
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in range(num_samples):
            line = f.readline()
            if not line:
                break
            samples.append(line)
    return samples


def calculate_compression_ratio(tokenizer, texts):
    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    total_tokens = 0
    for text in texts:
        total_tokens += len(tokenizer.encode(text))

    return total_bytes / total_tokens if total_tokens > 0 else 0


def measure_throughput(tokenizer, text_chunk):
    start_time = time.time()
    tokenizer.encode(text_chunk)
    end_time = time.time()

    elapsed = end_time - start_time
    num_bytes = len(text_chunk.encode("utf-8"))

    return num_bytes, elapsed, num_bytes / elapsed


def main():
    print("Loading tokenizers...")
    try:
        ts_tokenizer = load_tokenizer("tiny_stories")
        owt_tokenizer = load_tokenizer("owt")
    except FileNotFoundError as e:
        print(f"Error loading tokenizers: {e}")
        print("Please ensure training is complete.")
        return

    # Files
    ts_file = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"  # Use valid for sampling
    owt_file = DATA_DIR / "owt_valid.txt"

    print("\n--- (a) Compression Ratio ---")
    ts_samples = read_samples(ts_file, 10)
    owt_samples = read_samples(owt_file, 10)

    ts_ratio = calculate_compression_ratio(ts_tokenizer, ts_samples)
    owt_ratio = calculate_compression_ratio(owt_tokenizer, owt_samples)

    print(f"TinyStories (10k vocab) Ratio: {ts_ratio:.2f} bytes/token")
    print(f"OpenWebText (32k vocab) Ratio: {owt_ratio:.2f} bytes/token")

    print("\n--- (b) Cross-Tokenization (OWT samples with TS tokenizer) ---")
    cross_ratio = calculate_compression_ratio(ts_tokenizer, owt_samples)
    print(f"OWT samples with TS Tokenizer Ratio: {cross_ratio:.2f} bytes/token")
    print(f"Comparison: {'Worse' if cross_ratio < owt_ratio else 'Better'} compression than native OWT tokenizer.")
    print("Reasoning: TS tokenizer is optimized for simple stories, OWT has diverse web text.")

    print("\n--- (c) Throughput Estimation ---")
    # Read a larger chunk for throughput test
    large_chunk_size = 5 * 1024 * 1024  # 5MB
    with open(owt_file, "r") as f:
        large_chunk = f.read(large_chunk_size)

    bytes_processed, time_elapsed, throughput = measure_throughput(owt_tokenizer, large_chunk)
    print(f"Processed {bytes_processed / 1e6:.2f} MB in {time_elapsed:.2f} seconds")
    print(f"Throughput: {throughput / 1e6:.2f} MB/s")

    pile_size_gb = 825
    pile_size_bytes = pile_size_gb * 1024**3
    estimated_seconds = pile_size_bytes / throughput
    estimated_hours = estimated_seconds / 3600
    print(f"Estimated time for Pile (825GB): {estimated_hours:.2f} hours")

    print("\n--- (d) Encoding Datasets to uint16 ---")
    print("Why uint16? Because max vocab size is 32,000, which fits comfortably within uint16 (0-65535).")
    print("uint16 uses 2 bytes per token, saving 75% memory compared to int64 (8 bytes).")

    def encode_and_save(tokenizer, input_path, output_path, desc):
        print(f"Encoding {desc}...")
        start_time = time.time()

        def line_generator(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line

        ids = []
        # Progress tracking
        count = 0
        for token_id in tokenizer.encode_iterable(line_generator(input_path)):
            ids.append(token_id)
            count += 1
            if count % 1_000_000 == 0:
                print(f"  Processed {count / 1e6:.1f}M tokens...", end="\r")

        print(f"  Total tokens: {count}")

        # Convert to numpy uint16
        ids_np = np.array(ids, dtype=np.uint16)
        np.save(output_path, ids_np)

        end_time = time.time()
        print(f"  Saved to {output_path}")
        print(f"  Time taken: {end_time - start_time:.2f}s")
        return ids_np

    # Encode Validation Sets (Fast)
    try:
        ts_valid_out = OUTPUT_DIR / "tiny_stories_valid.npy"
        encode_and_save(ts_tokenizer, DATA_DIR / "TinyStoriesV2-GPT4-valid.txt", ts_valid_out, "TinyStories Valid")

        owt_valid_out = OUTPUT_DIR / "owt_valid.npy"
        encode_and_save(owt_tokenizer, DATA_DIR / "owt_valid.txt", owt_valid_out, "OpenWebText Valid")

        print("\nNote: Validation sets encoded. To encode full training sets, you can run similar commands.")
        # Uncomment to run full encoding if desired
        # encode_and_save(ts_tokenizer, DATA_DIR / "TinyStoriesV2-GPT4-train.txt", OUTPUT_DIR / "tiny_stories_train.npy", "TinyStories Train")
        # encode_and_save(owt_tokenizer, DATA_DIR / "owt_train.txt", OUTPUT_DIR / "owt_train.npy", "OpenWebText Train")

    except Exception as e:
        print(f"An error occurred during encoding: {e}")


if __name__ == "__main__":
    main()
