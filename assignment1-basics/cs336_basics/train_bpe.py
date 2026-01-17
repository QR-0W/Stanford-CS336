"""BPE 训练模块"""

import regex
from collections import Counter
from typing import BinaryIO


# GPT-2 风格的预分词正则表达式
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def count_pairs(word_freqs: dict[tuple, int]) -> Counter:
    """
    统计所有相邻字节对的频率

    Args:
        word_freqs: 单词到频率的映射

    Returns:
        pair_counts: 相邻字节对到频率的映射
    """
    pair_counts = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
    return pair_counts


def merge_word(word: tuple, pair: tuple, new_id: int) -> tuple:
    """
    在单个词中执行合并操作。

    Args:
        word: 单词的字节序列
        pair: 需要合并的字节对
        new_id: 新合并字节的 ID

    Returns:
        new_word: 合并后的单词字节序列
    """
    new_word = []
    i = 0
    while i < len(word):
        # 如果当前位置匹配要合并的对
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE 分词器。

    Args:
        input_path: 训练语料文件路径
        vocab_size: 目标词表大小（包括特殊 token）
        special_tokens: 特殊 token 列表

    Returns:
        vocab: Token ID 到字节串的映射
        merges: 合并规则列表
    """
    # step 1: 初始化词表（256 个字节 + 特殊 token）
    vocab = {i: bytes([i]) for i in range(256)}
    # 特殊 token 在开头加入，有自己的 token_id
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    merge = []

    # step 2：读取文件并预分词
    with open(input_path, "rb") as f:
        text = f.read()
    text_str = text.decode("utf-8", errors="ignore")

    # 先按特殊 token 分割文本，防止它们参与合并
    if special_tokens:
        # 使用 re.escape 确保特殊字符被正确转义
        special_pattern = "|".join(regex.escape(t) for t in special_tokens)
        # 分割文本，移除特殊 token 部分
        text_chunks = regex.split(special_pattern, text_str)
    else:
        text_chunks = [text_str]

    # 对每个 chunk 进行预分词
    tokens = []
    for chunk in text_chunks:
        tokens.extend(regex.findall(GPT2_PATTERN, chunk))

    # step 3: 统计字节对频率
    word_to_bytes = {}
    for token in tokens:
        token_bytes = tuple(token.encode("utf-8"))
        if token_bytes not in word_to_bytes:
            word_to_bytes[token_bytes] = 0
        word_to_bytes[token_bytes] += 1

    # step 4: 合并循环
    num_merges = vocab_size - 256 - len(special_tokens)
    for _ in range(num_merges):
        # 统计当前所有相邻对的频率
        pair_counts = count_pairs(word_to_bytes)

        # 没有可合并的对了
        if not pair_counts:
            break

        # 取频率最高的；频率相同时选字典序更大的那一对
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))

        # 创建新 token
        new_id = len(vocab)
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merge.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # 更新所有序列，执行合并
        new_word_to_bytes = {}
        for word, freq in word_to_bytes.items():
            new_word = merge_word(word, best_pair, new_id)
            if new_word not in new_word_to_bytes:
                new_word_to_bytes[new_word] = 0
            new_word_to_bytes[new_word] += freq
        word_to_bytes = new_word_to_bytes

    # 特殊 token 已经在开头加入，无需重复处理

    return vocab, merge
