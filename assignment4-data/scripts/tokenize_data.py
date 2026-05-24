"""用 GPT-2 tokenizer 将过滤后的文本转换为 ``np.uint16`` 训练二进制。

读取 ``filter_data.py`` 输出的 ``<|endoftext|>`` 分隔文本文件，
按分隔符拆分为独立文档，用 HuggingFace GPT-2 tokenizer 编码后
添加 ``eos_token_id``，最后展平为 ``np.uint16`` 数组并通过
``tofile()`` 序列化，供 ``cs336-basics/scripts/train.py`` 的
``np.memmap(..., dtype=np.uint16)`` 读取。

注意：多进程 worker 中需要设置 ``HF_HUB_OFFLINE=1`` 和
``TRANSFORMERS_OFFLINE=1`` 以避免子进程尝试通过代理访问
HuggingFace Hub 导致超时或 SOCKS 错误。
"""

from __future__ import annotations

import argparse
import gzip
import multiprocessing
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


ENDOFTEXT = "<|endoftext|>"


def _read_documents(input_paths: list[Path]) -> list[str]:
    """读取过滤后的文本文件，按 ``<|endoftext|>`` 分割为文档列表。

    支持 ``.gz`` 压缩文件和原始文本文件。
    """
    docs: list[str] = []
    for path in input_paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            content = f.read()
        for doc in content.split(ENDOFTEXT):
            doc = doc.strip()
            if doc:
                docs.append(doc)
    return docs


def _tokenize_document(doc: str, tokenizer: AutoTokenizer) -> list[int]:
    """对单篇文档做 GPT-2 编码并在末尾附加 EOS token。

    附加 EOS 的目的是为训练提供明确的 document boundary 信号，
    以便训练脚本在 batch 采样时正确跨越或跳过文档边界。
    """
    return tokenizer.encode(doc) + [tokenizer.eos_token_id]


def _tokenize_batch(docs: list[str], tokenizer_name: str) -> list[list[int]]:
    """批量 tokenize 文档（多进程 worker 入口）。

    关闭代理环境变量以避免子进程中的 SOCKS 连接错误，
    并设置 offline 模式防止访问 HuggingFace Hub。
    """
    import os as _os

    _os.environ["HF_HUB_OFFLINE"] = "1"
    _os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _os.environ.pop("HTTP_PROXY", None)
    _os.environ.pop("HTTPS_PROXY", None)
    _os.environ.pop("http_proxy", None)
    _os.environ.pop("https_proxy", None)
    _os.environ.pop("ALL_PROXY", None)
    _os.environ.pop("all_proxy", None)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    return [_tokenize_document(doc, tokenizer) for doc in docs]


def _chunks(lst: list[str], n: int):
    """将列表切分为长度为 n 的批次（最后一个批次可能不足 n）。"""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize filtered data with GPT-2 tokenizer and serialize as uint16 binary."
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="Filtered .txt or .txt.gz file(s) from filter_data.py.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .bin file for training (np.uint16).")
    parser.add_argument("--tokenizer", default="gpt2", help="HuggingFace tokenizer name.")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help="Number of parallel tokenizer processes.")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Documents per parallel batch.")
    args = parser.parse_args()

    print(f"Reading documents from {len(args.input)} input file(s)...")
    documents = _read_documents(args.input)
    print(f"Read {len(documents)} documents.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 多进程 tokenize，每个 batch 交给一个 worker
    all_ids: list[int] = []
    with multiprocessing.Pool(processes=args.workers) as pool:
        for batch_ids in tqdm(
            pool.starmap(_tokenize_batch, [(batch, args.tokenizer) for batch in _chunks(documents, args.chunk_size)]),
            total=len(documents) // args.chunk_size + (1 if len(documents) % args.chunk_size else 0),
            desc="Tokenizing documents",
        ):
            for doc_ids in batch_ids:
                all_ids.extend(doc_ids)

    # uint16 足以表示 GPT-2 vocab_size=50257（< 65535）
    ids_array = np.array(all_ids, dtype=np.uint16)
    print(f"Tokenized {len(documents)} documents into {len(all_ids)} tokens.")
    ids_array.tofile(str(args.output))
    print(f"Saved tokenized data to {args.output}")


if __name__ == "__main__":
    main()
