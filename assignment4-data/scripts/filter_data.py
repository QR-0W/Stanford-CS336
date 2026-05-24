from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import glob
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_data.harmful import classify_nsfw, classify_toxic_speech
from cs336_data.language import identify_language
from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.quality import classify_quality, gopher_quality_filter


_BLOCKED_URL_RE = re.compile(
    r"(?:sex|porn|casino|escort|adult|webcam|camto?cam|xxx|tgtg|wahas|haaxz)",
    re.IGNORECASE,
)
_LOW_VALUE_TEXT_RE = re.compile(
    r"(?:free\s+(?:live\s+)?sex\s+cams?|nude\s+webcam|private\s+cams?|"
    r"broadcast\s+yourself|casino|slot\s+machine|forum\s+registration\s+agreement|"
    r"log\s+in\s+to\s+check\s+your\s+private\s+messages)",
    re.IGNORECASE,
)
_NORMALIZE_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass
class FilterConfig:
    language: str = "en"
    language_threshold: float = 0.65
    quality_threshold: float = 0.55
    nsfw_threshold: float = 0.80
    toxic_threshold: float = 0.80
    min_chars: int = 200
    max_chars: int = 200_000
    use_quality_classifier: bool = True
    use_harmful_classifiers: bool = True
    mask_pii: bool = True
    sample_limit: int = 25
    sample_seed: int = 13


@dataclass
class FileStats:
    input_path: str
    output_path: str
    elapsed_seconds: float = 0.0
    counts: Counter[str] = field(default_factory=Counter)
    pii: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "elapsed_seconds": self.elapsed_seconds,
            "counts": dict(self.counts),
            "pii": dict(self.pii),
        }


def _read_warc_headers(f) -> tuple[dict[str, str] | None, int | None]:
    first = f.readline()
    if not first:
        return None, None
    while first in (b"\r\n", b"\n"):
        first = f.readline()
        if not first:
            return None, None

    headers = {"_status": first.decode("utf-8", errors="replace").strip()}
    while True:
        line = f.readline()
        if not line or line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if ":" in decoded:
            key, value = decoded.split(":", 1)
            headers[key] = value.strip()
    return headers, int(headers.get("Content-Length", "0"))


def iter_wet_records(path: Path, max_docs: int | None = None) -> Iterable[tuple[str, str]]:
    yielded = 0
    with gzip.open(path, "rb") as f:
        while max_docs is None or yielded < max_docs:
            headers, length = _read_warc_headers(f)
            if headers is None or length is None:
                break
            body = f.read(length)
            f.readline()
            f.readline()
            if headers.get("WARC-Type") != "conversion":
                continue
            text = body.decode("utf-8", errors="replace")
            yielded += 1
            yield headers.get("WARC-Target-URI", ""), text


def _normalize_for_lm(text: str) -> str:
    lines = [_NORMALIZE_SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def _mask_pii(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    text, count = mask_emails(text)
    counts["emails"] += count
    text, count = mask_phone_numbers(text)
    counts["phones"] += count
    text, count = mask_ips(text)
    counts["ips"] += count
    return text, counts


def _sample_record(samples: list[dict], sample: dict, limit: int, rng: random.Random, seen: int) -> None:
    if limit <= 0:
        return
    if len(samples) < limit:
        samples.append(sample)
        return
    index = rng.randint(0, seen - 1)
    if index < limit:
        samples[index] = sample


def _reject(reason: str, url: str, text: str, details: dict | None = None) -> tuple[bool, str, dict]:
    return False, reason, {"url": url, "text": text, "details": details or {}}


def filter_document(url: str, raw_text: str, config: FilterConfig) -> tuple[bool, str, str, Counter[str], dict]:
    text = _normalize_for_lm(raw_text)
    if not text:
        kept, reason, sample = _reject("empty", url, text)
        return kept, reason, text, Counter(), sample
    if len(text) < config.min_chars:
        kept, reason, sample = _reject("too_short", url, text, {"chars": len(text)})
        return kept, reason, text, Counter(), sample
    if len(text) > config.max_chars:
        kept, reason, sample = _reject("too_long", url, text, {"chars": len(text)})
        return kept, reason, text, Counter(), sample
    if _BLOCKED_URL_RE.search(url):
        kept, reason, sample = _reject("domain_blocklist", url, text)
        return kept, reason, text, Counter(), sample
    if _LOW_VALUE_TEXT_RE.search(text[:5000]):
        kept, reason, sample = _reject("low_value_pattern", url, text)
        return kept, reason, text, Counter(), sample

    language, language_score = identify_language(text)
    if language != config.language or language_score < config.language_threshold:
        kept, reason, sample = _reject(
            "language",
            url,
            text,
            {"language": language, "language_score": language_score},
        )
        return kept, reason, text, Counter(), sample

    if not gopher_quality_filter(text):
        kept, reason, sample = _reject("gopher", url, text)
        return kept, reason, text, Counter(), sample

    if config.use_harmful_classifiers:
        nsfw_label, nsfw_score = classify_nsfw(text)
        if nsfw_label == "nsfw" and nsfw_score >= config.nsfw_threshold:
            kept, reason, sample = _reject("nsfw", url, text, {"label": nsfw_label, "score": nsfw_score})
            return kept, reason, text, Counter(), sample

        toxic_label, toxic_score = classify_toxic_speech(text)
        if toxic_label == "toxic" and toxic_score >= config.toxic_threshold:
            kept, reason, sample = _reject("toxic", url, text, {"label": toxic_label, "score": toxic_score})
            return kept, reason, text, Counter(), sample

    if config.use_quality_classifier:
        quality_label, quality_score = classify_quality(text)
        if quality_label != "wiki" or quality_score < config.quality_threshold:
            kept, reason, sample = _reject(
                "quality_classifier",
                url,
                text,
                {"label": quality_label, "score": quality_score},
            )
            return kept, reason, text, Counter(), sample

    pii_counts: Counter[str] = Counter()
    if config.mask_pii:
        text, pii_counts = _mask_pii(text)

    return True, "kept", text, pii_counts, {"url": url, "text": text, "details": {}}


def process_wet_file(
    input_path: Path,
    output_path: Path,
    rejected_sample_path: Path,
    kept_sample_path: Path,
    config: FilterConfig,
    max_docs: int | None,
) -> FileStats:
    started = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_sample_path.parent.mkdir(parents=True, exist_ok=True)
    kept_sample_path.parent.mkdir(parents=True, exist_ok=True)

    stats = FileStats(str(input_path), str(output_path))
    rng = random.Random(f"{config.sample_seed}:{input_path}")
    kept_samples: list[dict] = []
    rejected_samples: list[dict] = []
    kept_seen = 0
    rejected_seen = 0

    with gzip.open(output_path, "wt", encoding="utf-8") as out:
        for url, raw_text in iter_wet_records(input_path, max_docs=max_docs):
            stats.counts["raw"] += 1
            kept, reason, text, pii_counts, sample = filter_document(url, raw_text, config)
            stats.counts[reason] += 1
            stats.pii.update(pii_counts)

            sample["reason"] = reason
            sample["excerpt"] = sample.pop("text")[:1200]
            if kept:
                kept_seen += 1
                _sample_record(kept_samples, sample, config.sample_limit, rng, kept_seen)
                out.write(text.replace("<|endoftext|>", " ").strip())
                out.write("\n<|endoftext|>\n")
            else:
                rejected_seen += 1
                _sample_record(rejected_samples, sample, config.sample_limit, rng, rejected_seen)

    with kept_sample_path.open("w", encoding="utf-8") as out:
        for sample in kept_samples:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with rejected_sample_path.open("w", encoding="utf-8") as out:
        for sample in rejected_samples:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    stats.elapsed_seconds = time.time() - started
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Common Crawl WET files into LM training text.")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Input .warc.wet.gz files or globs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(4, len(os.sched_getaffinity(0)))))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-docs-per-file", type=int, default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--language-threshold", type=float, default=0.65)
    parser.add_argument("--quality-threshold", type=float, default=0.55)
    parser.add_argument("--nsfw-threshold", type=float, default=0.80)
    parser.add_argument("--toxic-threshold", type=float, default=0.80)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--no-quality-classifier", action="store_true")
    parser.add_argument("--no-harmful-classifiers", action="store_true")
    parser.add_argument("--no-pii-masking", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=25)
    parser.add_argument("--sample-seed", type=int, default=13)
    return parser.parse_args()


def _expand_inputs(inputs: list[Path], max_files: int | None) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        matches = [Path(match) for match in sorted(glob.glob(str(input_path)))] if any(char in str(input_path) for char in "*?[]") else [input_path]
        paths.extend(path for path in matches if path.exists())
    deduped = sorted(dict.fromkeys(path.resolve() for path in paths))
    return deduped[:max_files] if max_files is not None else deduped


def main() -> None:
    args = _parse_args()
    input_paths = _expand_inputs(args.input, args.max_files)
    if not input_paths:
        raise SystemExit("No input WET files found.")

    config = FilterConfig(
        language=args.language,
        language_threshold=args.language_threshold,
        quality_threshold=args.quality_threshold,
        nsfw_threshold=args.nsfw_threshold,
        toxic_threshold=args.toxic_threshold,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        use_quality_classifier=not args.no_quality_classifier,
        use_harmful_classifiers=not args.no_harmful_classifiers,
        mask_pii=not args.no_pii_masking,
        sample_limit=args.sample_limit,
        sample_seed=args.sample_seed,
    )

    started = time.time()
    output_text_dir = args.output_dir / "text"
    kept_sample_dir = args.output_dir / "samples" / "kept"
    rejected_sample_dir = args.output_dir / "samples" / "rejected"
    stats_path = args.stats_output or args.output_dir / "filter_stats.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for input_path in input_paths:
            output_path = output_text_dir / f"{input_path.name}.filtered.txt.gz"
            kept_sample_path = kept_sample_dir / f"{input_path.name}.jsonl"
            rejected_sample_path = rejected_sample_dir / f"{input_path.name}.jsonl"
            futures.append(
                executor.submit(
                    process_wet_file,
                    input_path,
                    output_path,
                    rejected_sample_path,
                    kept_sample_path,
                    config,
                    args.max_docs_per_file,
                )
            )

        file_stats = [future.result() for future in concurrent.futures.as_completed(futures)]

    total_counts: Counter[str] = Counter()
    total_pii: Counter[str] = Counter()
    for stats in file_stats:
        total_counts.update(stats.counts)
        total_pii.update(stats.pii)

    elapsed = time.time() - started
    stats_payload = {
        "elapsed_seconds": elapsed,
        "num_input_files": len(input_paths),
        "workers": args.workers,
        "config": config.__dict__,
        "counts": dict(total_counts),
        "pii": dict(total_pii),
        "files": [stats.to_dict() for stats in sorted(file_stats, key=lambda item: item.input_path)],
    }
    with stats_path.open("w", encoding="utf-8") as out:
        json.dump(stats_payload, out, ensure_ascii=False, indent=2)

    raw = max(total_counts.get("raw", 0), 1)
    print(f"processed_files={len(input_paths)} elapsed_seconds={elapsed:.2f}")
    for key, value in total_counts.most_common():
        print(f"{key}\t{value}\t{value / raw:.4f}")
    print(f"stats={stats_path}")
    print(f"text_dir={output_text_dir}")


if __name__ == "__main__":
    main()
