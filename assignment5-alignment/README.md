# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## Zero-shot GSM8K baseline

Evaluate Qwen 2.5 Math 1.5B on the repository-provided GSM8K test split with the
R1-Zero prompt. This is the self-study replacement for the unavailable MATH split:

```sh
conda run -n Conf_Test python scripts/math_baseline.py \
  --model models/Qwen/Qwen2___5-Math-1___5B
```

By default, the script uses:

- Model: `models/Qwen/Qwen2___5-Math-1___5B` after downloading from ModelScope
- Data: `data/gsm8k/test.jsonl`
- Dataset format: `gsm8k` (extracts the short answer after `####`)
- Reward: numeric R1-Zero format reward
- Prompt: `cs336_alignment/prompts/r1_zero.prompt`
- Output: `outputs/math_baseline_gsm8k/`

For a small smoke test, pass `--limit`, for example:

```sh
conda run -n Conf_Test python scripts/math_baseline.py \
  --model models/Qwen/Qwen2___5-Math-1___5B \
  --limit 16
```

The script writes `generations.jsonl`, `metrics.json`, and
`category_examples.json` for the written analysis in `math_baseline`.

To run the original MATH setup on the course cluster, pass the handout paths and
the official reward function:

```sh
uv run python scripts/math_baseline.py \
  --model /data/a5-alignment/models/Qwen2.5-Math-1.5B \
  --data /data/a5-alignment/MATH/validation.jsonl \
  --dataset-format math \
  --reward-fn r1_zero \
  --output-dir outputs/math_baseline_math
```
