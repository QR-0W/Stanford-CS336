# Stanford CS336: Language Modeling from Scratch

This repository contains my solutions and implementations for Stanford's CS336 course assignments.

## Course Overview

CS336 focuses on building language models from scratch, covering:
- Tokenization (BPE, WordPiece)
- Transformer architecture implementation
- Training infrastructure
- Optimization techniques

## Repository Structure

```
CS336/
├── assignment1-basics/     # Assignment 1: Basics (Tokenizer + Transformer)
│   ├── cs336_basics/       # Implementation code
│   ├── tests/              # Unit tests
│   └── data/               # Training data (not tracked in git)
└── ...                     # Future assignments
```

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone this repository
git clone https://github.com/QR-0W/Stanford-CS336.git
cd Stanford-CS336

# Navigate to an assignment
cd assignment1-basics

# Download data
./download_data.sh

# Run tests
uv run pytest
```

## Assignments

### Assignment 1: Basics

- [x] Environment setup
- [ ] BPE Tokenizer implementation
- [ ] Transformer components (RMSNorm, RoPE, Attention)
- [ ] Language model training

## Resources

- [Official Course Repository](https://github.com/stanford-cs336/assignment1-basics)
- [Assignment Handout](assignment1-basics/cs336_spring2025_assignment1_basics.pdf)

## License

This repository is for educational purposes. Please refer to Stanford's academic integrity policies.

---

**Note**: This is a personal learning repository. Solutions are my own work for the CS336 course.
