"""训练 fastText wiki-vs-CC 质量分类器。

流程：
    1. 从 Wikipedia reference WARC 中提取正例文本（可选 Gopher 预过滤）
    2. 从 Common Crawl WET 中提取负例文本
    3. 写入 fastText 监督训练格式的文本文件
    4. 训练 fastText 模型并保存到 ``cs336_data/assets/quality_classifier.bin``

支持 ``--smoke-fixtures`` 先训练一个烟雾模型验证代码路径，
也支持 ``--apply-gopher`` 在正例采集阶段先过滤低质量页面。
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path
from typing import Iterable

import fasttext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_data.extraction import extract_text_from_html_bytes
from cs336_data.quality import gopher_quality_filter


LABEL_WIKI = "__label__wiki"
LABEL_CC = "__label__cc"


def _read_warc_headers(f):
    """解析 WARC record 头部，返回 (headers_dict, content_length)。"""
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
        s = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if ":" in s:
            k, v = s.split(":", 1)
            headers[k] = v.strip()
    return headers, int(headers.get("Content-Length", "0"))


def _normalize_doc(text: str) -> str:
    """将文本压缩为单行，避免 fastText 换行符导致训练格式错乱。"""
    return re.sub(r"\s+", " ", text).strip()


def _fasttext_line(label: str, text: str) -> str | None:
    """构造一条 fastText 监督训练样本行：``__label__xxx 文本内容``。"""
    text = _normalize_doc(text)
    if not text:
        return None
    return f"{label} {text}"


def iter_warc_texts(paths: Iterable[Path], max_docs: int | None = None) -> Iterable[str]:
    """迭代 WARC 文件中的 ``WARC-Type: response`` records，
    用 ``extract_text_from_html_bytes`` 抽取 HTML 正文。"""
    yielded = 0
    for path in paths:
        with gzip.open(path, "rb") as f:
            while max_docs is None or yielded < max_docs:
                headers, length = _read_warc_headers(f)
                if headers is None:
                    break
                body = f.read(length)
                f.readline(); f.readline()
                if headers.get("WARC-Type") != "response":
                    continue
                # HTTP response body 在 WARC 中格式为 "headers\r\n\r\nbody"
                html_body = body.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in body else body
                text = extract_text_from_html_bytes(html_body) or ""
                if text.strip():
                    yielded += 1
                    yield text


def iter_wet_texts(paths: Iterable[Path], max_docs: int | None = None) -> Iterable[str]:
    """迭代 WET 文件中的 ``WARC-Type: conversion`` records，直接读预抽取文本。"""
    yielded = 0
    for path in paths:
        with gzip.open(path, "rb") as f:
            while max_docs is None or yielded < max_docs:
                headers, length = _read_warc_headers(f)
                if headers is None:
                    break
                body = f.read(length)
                f.readline(); f.readline()
                if headers.get("WARC-Type") != "conversion":
                    continue
                text = body.decode("utf-8", errors="replace")
                if text.strip():
                    yielded += 1
                    yield text


def iter_fixture_examples(fixtures_dir: Path) -> Iterable[tuple[str, str]]:
    """从测试 fixtures 中构造少量样本用于 smoke 训练。"""
    high_quality = fixtures_dir / "high_quality_wiki_reference.txt"
    low_quality = fixtures_dir / "low_quality_cc.txt"

    wiki_text = high_quality.read_text(encoding="utf-8")
    for para in re.split(r"\n\s*\n", wiki_text):
        if len(para.split()) >= 30:
            yield LABEL_WIKI, para
    yield LABEL_WIKI, wiki_text

    cc_text = low_quality.read_text(encoding="utf-8")
    yield LABEL_CC, cc_text
    for line in cc_text.splitlines():
        if line.strip():
            yield LABEL_CC, line


def build_train_file(args: argparse.Namespace) -> None:
    """根据命令行参数构造训练数据文件。"""
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    with args.train_output.open("w", encoding="utf-8") as out:
        if args.smoke_fixtures:
            for label, text in iter_fixture_examples(args.fixtures_dir):
                line = _fasttext_line(label, text)
                if line:
                    out.write(line + "\n")
            return

        for text in iter_warc_texts(args.positive_warcs, args.max_positive_docs):
            # 正例需要做 Gopher 预过滤以提高正例质量
            if not args.apply_gopher or gopher_quality_filter(text):
                line = _fasttext_line(LABEL_WIKI, text)
                if line:
                    out.write(line + "\n")

        for text in iter_wet_texts(args.negative_wets, args.max_negative_docs):
            line = _fasttext_line(LABEL_CC, text)
            if line:
                out.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fastText wiki-vs-cc quality classifier.")
    parser.add_argument("--positive-warcs", type=Path, nargs="*", default=[])
    parser.add_argument("--negative-wets", type=Path, nargs="*", default=[])
    parser.add_argument("--train-output", type=Path, default=Path("data/quality_classifier.train.txt"))
    parser.add_argument("--model-output", type=Path, default=Path("cs336_data/assets/quality_classifier.bin"))
    parser.add_argument("--max-positive-docs", type=int, default=None)
    parser.add_argument("--max-negative-docs", type=int, default=None)
    parser.add_argument("--apply-gopher", action="store_true")
    parser.add_argument("--smoke-fixtures", action="store_true")
    parser.add_argument("--fixtures-dir", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--word-ngrams", type=int, default=2)
    args = parser.parse_args()

    build_train_file(args)
    # fastText 监督训练
    model = fasttext.train_supervised(
        input=str(args.train_output),
        epoch=args.epoch,
        lr=args.lr,
        wordNgrams=args.word_ngrams,
        minCount=1,
        verbose=0,
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_output))
    print(f"wrote train data to {args.train_output}")
    print(f"wrote model to {args.model_output}")


if __name__ == "__main__":
    main()
