from __future__ import annotations

import hashlib
import os
import re
import shutil
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    line_counts: Counter[str] = Counter()
    for file_path in input_files:
        with open(file_path, encoding="utf-8") as f:
            line_counts.update(f.readlines())

    for file_path in input_files:
        file_path = Path(file_path)
        with open(file_path, encoding="utf-8") as f:
            unique_lines = [line for line in f.readlines() if line_counts[line] == 1]
        with open(output_path / file_path.name, "w", encoding="utf-8") as out:
            out.writelines(unique_lines)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _word_ngrams(text: str, n: int) -> set[str]:
    tokens = _normalize_text(text).split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _hash_int(value: str, seed: int) -> int:
    payload = f"{seed}\0{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _minhash_signature(ngrams: set[str], num_hashes: int) -> tuple[int, ...]:
    if not ngrams:
        return tuple(0 for _ in range(num_hashes))
    return tuple(min(_hash_int(ngram, seed) for ngram in ngrams) for seed in range(num_hashes))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if root_a < root_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_a] = root_b


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = [Path(path) for path in input_files]
    texts = [path.read_text(encoding="utf-8") for path in paths]
    shingle_sets = [_word_ngrams(text, ngrams) for text in texts]
    signatures = [_minhash_signature(shingles, num_hashes) for shingles in shingle_sets]

    band_size = max(1, num_hashes // num_bands)
    candidates: set[tuple[int, int]] = set()
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for doc_idx, signature in enumerate(signatures):
        for band_idx in range(num_bands):
            start = band_idx * band_size
            end = num_hashes if band_idx == num_bands - 1 else min(num_hashes, start + band_size)
            if start >= num_hashes:
                break
            key = (band_idx, signature[start:end])
            for other_idx in buckets[key]:
                candidates.add((other_idx, doc_idx))
            buckets[key].append(doc_idx)

    uf = _UnionFind(len(paths))
    for i, j in candidates:
        if _jaccard(shingle_sets[i], shingle_sets[j]) >= jaccard_threshold:
            uf.union(i, j)

    kept_roots: set[int] = set()
    for idx, path in enumerate(paths):
        root = uf.find(idx)
        if root in kept_roots:
            continue
        kept_roots.add(root)
        shutil.copyfile(path, output_path / path.name)
