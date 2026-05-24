from __future__ import annotations

import argparse
import gzip
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Wikipedia reference URLs for quality-classifier positives.")
    parser.add_argument("--input", type=Path, required=True, help="Path to enwiki extracted URLs .txt.gz")
    parser.add_argument("--output", type=Path, required=True, help="Output sampled URL text file")
    parser.add_argument("--num-urls", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()

    random.seed(args.seed)
    reservoir: list[str] = []
    seen = 0
    with gzip.open(args.input, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            url = line.strip()
            if not url or not (url.startswith("http://") or url.startswith("https://")):
                continue
            seen += 1
            if len(reservoir) < args.num_urls:
                reservoir.append(url)
            else:
                j = random.randrange(seen)
                if j < args.num_urls:
                    reservoir[j] = url

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(reservoir) + "\n", encoding="utf-8")
    print(f"sampled {len(reservoir)} URLs from {seen} valid URLs into {args.output}")


if __name__ == "__main__":
    main()
