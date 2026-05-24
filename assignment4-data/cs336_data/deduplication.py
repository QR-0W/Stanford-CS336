"""文档去重：行级精确去重 + 文档级 MinHash/LSH 近似去重。

提供两个层次的去重功能：

1. ``exact_line_deduplication``
   全局行级精确匹配去重。任何一行在全部输入文件中出现超过一次，
   则该行从所有文件中删除。

2. ``minhash_deduplication``
   文档级近似去重。流程：文本规范化 -> word n-gram shingles ->
   MinHash 签名 -> LSH 候选对生成 -> 真实 Jaccard 验证 ->
   Union-Find 聚类 -> 每簇保留一个代表文档。
"""

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
    """行级精确去重：删除在所有文件中出现超过一次的行。

    两遍扫描：
        1. 统计全局行频。
        2. 重写每个文件，只保留全局频次为 1 的行。

    注意：行比较是字节/字符串级别的 exact match，不做 normalization。
    大小写、标点、空白的差异都会导致两行被视为不同行。

    Args:
        input_files: 待去重的文本文件路径列表。
        output_directory: 输出去重后文件的目录，保持原 basename。
    """
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # 第一遍：全局行频统计
    line_counts: Counter[str] = Counter()
    for file_path in input_files:
        with open(file_path, encoding="utf-8") as f:
            line_counts.update(f.readlines())

    # 第二遍：按文件重写，跳过重复行
    for file_path in input_files:
        file_path = Path(file_path)
        with open(file_path, encoding="utf-8") as f:
            unique_lines = [line for line in f.readlines() if line_counts[line] == 1]
        with open(output_path / file_path.name, "w", encoding="utf-8") as out:
            out.writelines(unique_lines)


def _normalize_text(text: str) -> str:
    """MinHash 前的文本规范化。

    步骤：NFD 规范化 -> 去除重音符号 -> 小写 -> 去标点 -> 合并空白。
    这样处理后，仅版权行/空白/标点/重音不同的近似文档会被发现。
    """
    text = unicodedata.normalize("NFD", text)
    # 去除 combining marks（重音、变音符号等）
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _word_ngrams(text: str, n: int) -> set[str]:
    """从规范化文本中提取 word n-gram shingle 集合。

    若文本 token 数不足 n，返回全部 token 拼接的单个 shingle。
    """
    tokens = _normalize_text(text).split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _hash_int(value: str, seed: int) -> int:
    """用 seed 对字符串做 BLAKE2b 64-bit 哈希，返回无符号整数。

    用于 MinHash：不同 seed 对应不同的"哈希函数"。
    """
    payload = f"{seed}\0{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _minhash_signature(ngrams: set[str], num_hashes: int) -> tuple[int, ...]:
    """计算 n-gram 集合的 MinHash 签名。

    对每个 seed i∈[0, num_hashes)，取所有 n-gram 在该 seed 下哈希的最小值。
    签名长度 = num_hashes。

    空集合返回全 0 签名。
    """
    if not ngrams:
        return tuple(0 for _ in range(num_hashes))
    return tuple(min(_hash_int(ngram, seed) for ngram in ngrams) for seed in range(num_hashes))


def _jaccard(a: set[str], b: set[str]) -> float:
    """计算两个集合的 Jaccard 相似度。

    处理边界：两个都空 -> 1.0；一个空 -> 0.0。
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UnionFind:
    """并查集，用于将 LSH 候选对合并为去重簇。

    使用路径压缩 (path compression) 优化 find。
    """

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        """查找根节点（带路径压缩）。"""
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        """合并 a 和 b 所在的集合（按根节点索引大小决定方向）。"""
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
    """MinHash + LSH 文档级近似去重。

    实现流程：
        1. 读取并规范化文档文本。
        2. 提取 word n-gram shingle 集合。
        3. 计算 MinHash 签名。
        4. 用 LSH banding 生成候选重复对（不直接比较所有文档对）。
        5. 对候选对计算真实 Jaccard 相似度。
        6. 相似度 >= ``jaccard_threshold`` 的对用并查集合并。
        7. 每个簇保留输入顺序最靠前的一个文档。

    LSH 仅用于候选生成，最终去重决策仍基于真实 Jaccard 相似度，
    因此不会因单一 band 碰撞误删文档。

    Args:
        input_files: 待去重文档路径列表。
        num_hashes: MinHash 签名维数。
        num_bands: LSH band 数。
        ngrams: word n-gram 长度。
        jaccard_threshold: 判定为重复的最低 Jaccard 相似度。
        output_directory: 输出去重后文件的目录。
    """
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1-2: 读取文本并构造 shingle 集合
    paths = [Path(path) for path in input_files]
    texts = [path.read_text(encoding="utf-8") for path in paths]
    shingle_sets = [_word_ngrams(text, ngrams) for text in texts]

    # 3: 计算 MinHash 签名
    signatures = [_minhash_signature(shingles, num_hashes) for shingles in shingle_sets]

    # 4: LSH banding -> 候选对
    # 将签名切成 num_bands 个 band，每个 band 内签名相同的文档视为候选
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
            # 同一 bucket 内已有的文档都是候选
            for other_idx in buckets[key]:
                candidates.add((other_idx, doc_idx))
            buckets[key].append(doc_idx)

    # 5-6: True Jaccard 验证 + Union-Find 聚类
    uf = _UnionFind(len(paths))
    for i, j in candidates:
        if _jaccard(shingle_sets[i], shingle_sets[j]) >= jaccard_threshold:
            uf.union(i, j)

    # 7: 输出，每个簇只保留第一个出现根节点的文档
    kept_roots: set[int] = set()
    for idx, path in enumerate(paths):
        root = uf.find(idx)
        if root in kept_roots:
            continue
        kept_roots.add(root)
        shutil.copyfile(path, output_path / path.name)
