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
    """
    将文件分割成可独立处理的块，边界对齐到 special token。

    用于多进程并行处理大文件时，确保每个块在语义边界处分割（如文档边界）。

    Args:
        file: 已打开的二进制文件对象
        desired_num_chunks: 期望的块数量（通常等于进程数）
        split_special_token: 用于对齐边界的特殊 token（如 endoftext）

    Returns:
        list[int]: 块边界的字节偏移列表，长度为 desired_num_chunks + 1
                   例如 [0, 1000, 2000, 3000] 表示 3 个块
    """
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
    """
    处理单个文件块，返回词频 Counter。

    这是多进程并行预分词的工作函数，每个进程独立处理一个文件块。

    Args:
        args: 包含 (input_path, start, end, special_tokens) 的元组
            - input_path (str): 输入文件路径
            - start (int): 块的起始字节偏移
            - end (int): 块的结束字节偏移
            - special_tokens (list[str]): 特殊 token 列表

    Returns:
        Counter: 词频计数器，键为 tuple[int]（字节序列），值为出现次数
    """
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
    """
    在单个词中执行合并操作。

    将词中所有出现的相邻 token 对 (pair[0], pair[1]) 替换为单个新 token (new_id)。

    Args:
        word: 词的 token 序列，例如 (72, 101, 108, 108, 111) 表示 "hello"
        pair: 要合并的 token 对，例如 (108, 108) 表示 "ll"
        new_id: 合并后的新 token ID

    Returns:
        tuple: 合并后的新 token 序列

    Example:
        >>> merge_word((72, 101, 108, 108, 111), (108, 108), 256)
        (72, 101, 256, 111)  # "he" + "ll" + "o" -> "he" + [new_token] + "o"
    """
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

    为增量更新优化做准备，建立 pair -> words 的反向索引。

    Args:
        word_freqs: 词频字典，键为 tuple[int]（词的 token 序列），值为出现次数

    Returns:
        tuple: 包含两个元素的元组
            - pair_counts (Counter): 每个 token 对的总频率
            - pair_to_words (dict): 反向索引，记录每个 pair 出现在哪些词中
                                   键为 tuple (pair)，值为 set (包含该 pair 的词集合)
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

    只更新与合并对相关的词，而不是重新遍历所有词。这是 BPE 训练速度优化的关键。

    工作流程：
    1. 找到所有包含 best_pair 的词（通过反向索引）
    2. 对每个受影响的词：
       - 删除旧词的所有 pair 计数
       - 执行合并得到新词
       - 添加新词的 pair 计数
       - 更新反向索引

    Args:
        word_freqs: 词频字典（会被原地修改）
        pair_counts: pair 频率计数器（会被原地修改）
        pair_to_words: pair 到词的反向索引（会被原地修改）
        best_pair: 要合并的 token 对
        new_id: 合并后的新 token ID
        vocab: 词表字典（用于 tie-breaking，不会被修改）

    Returns:
        None: 所有修改都是原地进行的
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

    实现了两个关键优化：
    1. **多进程并行预分词**: 将大文件分块，使用多个进程并行处理
    2. **增量对计数更新**: 每次合并只更新受影响的词，避免全量重新统计

    训练流程：
    1. 初始化 256 个字节 token + 特殊 token
    2. 并行预分词：将文本切分成 pre-token 并统计词频
    3. 初始化 pair 计数和反向索引
    4. 迭代合并：
       - 找频率最高的 pair（频率相同时按字典序）
       - 创建新 token
       - 增量更新受影响词的 pair 计数

    Args:
        input_path: 训练语料文件路径（支持大文件，如 12GB）
        vocab_size: 目标词表大小（包括 256 个基础字节 + 特殊 token）
        special_tokens: 特殊 token 列表（如 ["&lt;|endoftext|&gt;"]），这些 token 不会被拆分
        num_workers: 并行进程数，默认为 min(cpu_count(), 8)

    Returns:
        tuple: 包含两个元素的元组
            - vocab (dict[int, bytes]): Token ID 到字节串的映射
            - merges (list[tuple[bytes, bytes]]): 合并规则列表，按创建顺序排列

    Example:
        >>> vocab, merges = train_bpe("data.txt", 10000, ["<|endoftext|>"])
        >>> print(len(vocab))  # 10000
        >>> print(vocab[256])  # b'<|endoftext|>'
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
