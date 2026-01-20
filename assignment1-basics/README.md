# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment

We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using

```sh
uv run <python_file_path>
```

and the environment will be automatically solved and activated when necessary.

### Run unit tests

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data

Download the TinyStories data and a subsample of OpenWebText

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

````

## Implementation Status (Part 1: BPE)

- [x] **BPE Training (`train_bpe.py`)**: Implemented memory-efficient parallel BPE training.
- [x] **Tokenizer Class (`tokenizer.py`)**: Implemented encoding (with GPT-2 pre-tokenization), decoding, and file loading.
- [x] **Optimization**: Achieved significantly reduced memory usage for large corpora via streaming and multiprocessing.
- [x] **Evaluation**:
  - TinyStories (10k vocab) Compression Ratio: **4.03 bytes/token**
  - OpenWebText (32k vocab) Compression Ratio: **4.70 bytes/token** (on OWT)
  - Throughput: **~1.66 MB/s**

### Usage

**Train Tokenizer:**
```sh
uv run cs336_basics/train_tinystories_bpe.py
# or
uv run cs336_basics/train_owt_bpe.py
````

**Evaluate:**

```sh
uv run cs336_basics/evaluate_tokenizer_qa.py
```
