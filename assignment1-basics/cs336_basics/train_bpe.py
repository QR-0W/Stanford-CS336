"""BPE 训练模块 - 内存与速度优化版"""

import os
import regex
from collections import Counter, defaultdict
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
import heapq


# GPT-2 风格的预分词正则表达式
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """将文件分割成可独立处理的块，边界对齐到 special token。"""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def process_chunk(args: tuple) -> Counter:
    """处理单个文件块，返回词频 Counter。"""
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    if special_tokens:
        special_pattern = "|".join(regex.escape(t) for t in special_tokens)
        text_parts = regex.split(special_pattern, chunk_text)
    else:
        text_parts = [chunk_text]

    word_freqs = Counter()
    for part in text_parts:
        for token in regex.findall(GPT2_PATTERN, part):
            token_bytes = tuple(token.encode("utf-8"))
            word_freqs[token_bytes] += 1

    return word_freqs


def merge_word(word: tuple, pair: tuple, new_id: int) -> tuple:
    """在单个词中执行合并操作"""
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def get_pair_counts(word_freqs: dict[tuple, int]) -> tuple[Counter, dict]:
    """
    初始化对计数和反向索引。
    返回: (pair_counts, pair_to_words)
    - pair_counts: 每个对的总频率
    - pair_to_words: 每个对出现在哪些词中
    """
    pair_counts = Counter()
    pair_to_words = defaultdict(set)

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    return pair_counts, pair_to_words


def update_pair_counts_after_merge(
    word_freqs: dict[tuple, int],
    pair_counts: Counter,
    pair_to_words: dict,
    best_pair: tuple,
    new_id: int,
    vocab: dict,
) -> None:
    """
    增量更新对计数（核心优化）。
    只更新与合并对相关的词，而不是重新遍历所有词。
    """
    # 找到包含这个对的所有词
    affected_words = list(pair_to_words.get(best_pair, set()))

    for word in affected_words:
        if word not in word_freqs:
            continue
        freq = word_freqs[word]

        # 1. 删除旧词的所有对计数
        for i in range(len(word) - 1):
            old_pair = (word[i], word[i + 1])
            pair_counts[old_pair] -= freq
            if pair_counts[old_pair] <= 0:
                del pair_counts[old_pair]
            pair_to_words[old_pair].discard(word)

        # 2. 执行合并得到新词
        new_word = merge_word(word, best_pair, new_id)

        # 3. 更新 word_freqs
        del word_freqs[word]
        if new_word in word_freqs:
            word_freqs[new_word] += freq
        else:
            word_freqs[new_word] = freq

        # 4. 添加新词的对计数
        for i in range(len(new_word) - 1):
            new_pair = (new_word[i], new_word[i + 1])
            pair_counts[new_pair] += freq
            pair_to_words[new_pair].add(new_word)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE 分词器（优化版）。

    优化点：
    1. 多进程并行预分词
    2. 增量更新对计数（不重新遍历所有词）
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    # step 1: 初始化词表
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    merges = []

    # step 2: 分块并行预分词
    print(f"使用 {num_workers} 个进程进行并行预分词...")

    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n\n"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, split_token)

    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool(num_workers) as pool:
        chunk_results = pool.map(process_chunk, chunk_args)

    # 合并所有 chunk 的词频
    word_freqs = Counter()
    for chunk_freqs in chunk_results:
        word_freqs.update(chunk_freqs)

    # 转换为普通 dict 以便修改
    word_freqs = dict(word_freqs)

    print(f"预分词完成，共 {len(word_freqs)} 个唯一词")

    # step 3: 初始化对计数和索引
    print("初始化对计数...")
    pair_counts, pair_to_words = get_pair_counts(word_freqs)
    print(f"初始对数量: {len(pair_counts)}")

    # step 4: 合并循环（使用增量更新）
    num_merges = vocab_size - 256 - len(special_tokens)
    print(f"开始合并，目标 {num_merges} 次...")

    for step in range(num_merges):
        if not pair_counts:
            print(f"没有更多可合并的对，在第 {step} 步停止")
            break

        # 找频率最高的对
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))

        # 创建新 token
        new_id = len(vocab)
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # 增量更新对计数
        update_pair_counts_after_merge(word_freqs, pair_counts, pair_to_words, best_pair, new_id, vocab)

        # 进度打印
        if (step + 1) % 1000 == 0:
            print(f"合并进度: {step + 1}/{num_merges}, 剩余唯一对: {len(pair_counts)}")

    print(f"合并完成，词表大小: {len(vocab)}")
    return vocab, merges
