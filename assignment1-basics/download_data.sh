#!/bin/bash

# Create data directory and enter it
mkdir -p data
cd data

# Download TinyStories dataset
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# Download and extract OWT sample dataset
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip -f owt_train.txt.gz

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip -f owt_valid.txt.gz

# Go back to previous directory
cd ..
