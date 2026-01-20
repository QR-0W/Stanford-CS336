"""BPE 分词器实现"""

import json
import regex
from collections.abc import Iterator, Iterable


# GPT-2 风格的预分词正则表达式
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """BPE 分词器类"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        构造分词器。

        Args:
            vocab: Token ID 到字节串的映射
            merges: 合并规则列表
            special_tokens: 特殊 token 列表（可选）
        """
        # 保存 vocab 和 merges
        self.vocab = dict(vocab)
        self.merges = list(merges)

        # 处理特殊 token
        self.special_tokens = special_tokens or []
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in set(self.vocab.values()):
                self.vocab[len(self.vocab)] = token_bytes

        # 建 bytes -> id 的反向映射（用于 encode）
        self.byte_to_id = {v: k for k, v in vocab.items()}

        # 构建 merge 优先级字典（用于快速查找）
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        """
        从文件加载词表和合并规则，构造 Tokenizer。

        Args:
            vocab_filepath: 词表文件路径 (JSON 格式)
            merges_filepath: 合并规则文件路径 (JSON 格式)
            special_tokens: 特殊 token 列表（可选）

        Returns:
            Tokenizer: 构造好的 Tokenizer 实例
        """
        # 1. 加载 Vocab (json)
        # 格式: {"0": "hex_string", ...}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        vocab = {int(k): bytes.fromhex(v) for k, v in vocab_data.items()}

        # 2. 加载 Merges (json)
        # 格式: [["hex1", "hex2"], ...]
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_data = json.load(f)

        merges = [(bytes.fromhex(p[0]), bytes.fromhex(p[1])) for p in merges_data]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为 token ID 列表。

        Args:
            text: 待编码的文本

        Returns:
            list[int]: 编码后的 token ID 列表
        """
        # 实现编码逻辑
        if not text:
            return []

        result = []

        # 1. 先按照特殊 token 预分词
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=lambda x: len(x), reverse=True)
            special_pattern = "(" + "|".join(regex.escape(t) for t in sorted_special_tokens) + ")"
            parts = regex.split(special_pattern, text)
        else:
            parts = [text]

        # 2. 对每个 part 进行预分词
        for part in parts:
            if not part:
                continue

            # 如果是特殊 token，直接添加
            if self.special_tokens:
                token_bytes = part.encode("utf-8")
                if token_bytes in self.byte_to_id:
                    result.append(self.byte_to_id[token_bytes])
                    continue

            # 否则使用 GPT-2 预分词
            pre_tokens = regex.findall(GPT2_PATTERN, part)

            for pre_token in pre_tokens:
                # 3. 转成字节列表（每个字节是一个初始 token）
                token_bytes_list = [bytes([b]) for b in pre_token.encode("utf-8")]

                # 4. BPE 合并：反复合并优先级最高的相邻对
                while len(token_bytes_list) > 1:
                    # 找所有相邻对中优先级最高的
                    best_pair = None
                    best_priority = float("inf")
                    best_idx = -1

                    for i in range(len(token_bytes_list) - 1):
                        pair = (token_bytes_list[i], token_bytes_list[i + 1])
                        if pair in self.merge_priority:
                            if self.merge_priority[pair] < best_priority:
                                best_priority = self.merge_priority[pair]
                                best_pair = pair
                                best_idx = i

                    # 没有可合并的对了
                    if best_pair is None:
                        break

                    # 执行合并
                    merged = token_bytes_list[best_idx] + token_bytes_list[best_idx + 1]
                    token_bytes_list = token_bytes_list[:best_idx] + [merged] + token_bytes_list[best_idx + 2 :]

                # 5. 查表得到 token ID
                for token_bytes in token_bytes_list:
                    if token_bytes in self.byte_to_id:
                        result.append(self.byte_to_id[token_bytes])

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        内存高效的流式编码。

        Args:
            iterable: 包含字符串的可迭代对象

        Yields:
            int: 编码后的 token ID
        """
        # 实现流式编码
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 列表解码为文本。

        Args:
            ids: token ID 列表

        Returns:
            str: 解码后的文本

        Raises:
            ValueError: 如果 token ID 不在词表中
        """
        # 实现解码逻辑
        # 1. id -> bytes
        byte_list = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_list.append(self.vocab[token_id])
            else:
                raise ValueError(f"Token ID {token_id} not found in vocab")

        # 2. 拼接 bytes
        all_bytes = b"".join(byte_list)

        # 3. bytes -> str
        return all_bytes.decode("utf-8", errors="replace")
