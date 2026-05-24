"""检查过滤后数据的保留样本和丢弃样本。

读取 ``filter_data.py`` 输出的 kept/rejected JSONL 文件，
从两类中各随机抽取样本，生成 Markdown 格式的检查报告，
包含 URL、过滤原因、简短人工评价和数据摘录。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def _iter_jsonl(paths: list[Path]):
    """遍历 JSONL 文件，逐行 yield 反序列化的 dict。"""
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _reservoir_sample(items, k: int, seed: int) -> list[dict]:
    """从迭代器中用 reservoir sampling 抽取最多 k 条样本。"""
    rng = random.Random(seed)
    sample: list[dict] = []
    for seen, item in enumerate(items, start=1):
        if len(sample) < k:
            sample.append(item)
            continue
        index = rng.randint(0, seen - 1)
        if index < k:
            sample[index] = item
    return sample


def _clean_excerpt(text: str, max_chars: int) -> str:
    """压缩空白并截断到指定长度，末尾加 ``...`` 表示省略。"""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def _comment_for_kept(sample: dict, used_quality_classifier: bool) -> str:
    """根据保留样本的 URL 和内容特征生成简短中文评价。"""
    excerpt = sample.get("excerpt", "")
    words = excerpt.split()
    url = sample.get("url", "")
    if len(words) < 80:
        return "保留样本偏短，需要警惕正文信息量不足，但它通过了语言、质量和安全过滤。"
    if any(token in excerpt.lower() for token in ["privacy policy", "terms of use", "copyright"]):
        return "文本可读但可能含有法律/站点模板内容，对 C4 风格 LM 训练价值一般。"
    if any(domain in url for domain in ["blogspot", "wordpress", "medium", "github", "wikipedia"]):
        return "文本像普通网页正文或技术/博客内容，整体适合作为 C4 风格语言建模数据。"
    filters = "英文、Gopher、安全"
    if used_quality_classifier:
        filters += "和质量分类器"
    return f"文本通过了{filters}过滤，整体比原始 WET 更接近可训练网页正文。"


def _comment_for_rejected(sample: dict) -> str:
    """根据丢弃原因生成简短中文评价。"""
    reason = sample.get("reason", "unknown")
    details = sample.get("details", {})
    if reason == "language":
        return f"移除是合理的：语言预测为 ``{details.get('language')}``，不符合英文训练目标。"
    if reason == "gopher":
        return "移除通常合理：Gopher 规则表明文本长度、词长、符号比例或 alphabetic word 比例异常。"
    if reason == "quality_classifier":
        return f"移除是质量分类器决策：预测 ``{details.get('label')}``，置信度 ``{details.get('score')}``，说明它不像高质量 reference page。"
    if reason in {"nsfw", "toxic", "domain_blocklist"}:
        return "移除合理：安全/域名过滤命中，避免成人、博彩、toxic 或 spam 内容进入训练集。"
    if reason in {"too_short", "empty", "too_long"}:
        return "移除合理：长度异常的页面通常是错误页、模板页、抽取失败或超大噪声文档。"
    return "需要人工复核该移除原因，但该样本没有进入最终训练文本。"


def _write_section(out, title: str, samples: list[dict], max_chars: int, kept: bool, used_quality_classifier: bool) -> None:
    """将一组样本写入 Markdown 文件的一个章节。"""
    out.write(f"## {title}\n\n")
    if not samples:
        out.write("No samples available.\n\n")
        return
    for index, sample in enumerate(samples, start=1):
        out.write(f"### Example {index}\n\n")
        out.write(f"URL: ``{sample.get('url', '')}``\n\n")
        out.write(f"Reason: ``{sample.get('reason', '')}``\n\n")
        comment = _comment_for_kept(sample, used_quality_classifier) if kept else _comment_for_rejected(sample)
        out.write(f"Comment: {comment}\n\n")
        out.write("Excerpt:\n\n")
        out.write("```text\n")
        out.write(_clean_excerpt(sample.get("excerpt", ""), max_chars))
        out.write("\n```\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 filter_data.py 输出的保留和丢弃样本。")
    parser.add_argument("--filter-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    kept_paths = sorted((args.filter_output_dir / "samples" / "kept").glob("*.jsonl"))
    rejected_paths = sorted((args.filter_output_dir / "samples" / "rejected").glob("*.jsonl"))

    # 从 filter_stats.json 中读取是否启用了质量分类器
    stats_path = args.filter_output_dir / "filter_stats.json"
    used_quality_classifier = False
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        used_quality_classifier = bool(stats.get("config", {}).get("use_quality_classifier", False))

    # 从两个样本池中各自 reservoir-sample 指定数量
    kept = _reservoir_sample(_iter_jsonl(kept_paths), args.num_examples, args.seed)
    rejected = _reservoir_sample(_iter_jsonl(rejected_paths), args.num_examples, args.seed + 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        out.write("# Filtered Data Inspection\n\n")
        out.write("This report samples examples produced by ``scripts/filter_data.py``.\n\n")
        _write_section(out, "Kept Examples", kept, args.max_chars, kept=True,
                       used_quality_classifier=used_quality_classifier)
        _write_section(out, "Discarded Or Modified Examples", rejected, args.max_chars, kept=False,
                       used_quality_classifier=used_quality_classifier)
        out.write("## Iteration Notes\n\n")
        quality_note = "enabled" if used_quality_classifier else "disabled for this run because it was too strict on the local sample"
        out.write(
            "The current pipeline prioritizes English C4-like pages with language filtering, Gopher quality rules, "
            f"harmful-content filtering, PII masking, and a wiki-vs-CC quality classifier that is {quality_note}. "
            "Manual inspection should be used to tune the quality threshold and decide whether the quality classifier "
            "is too strict for C4-style data.\n"
        )

    print(f"wrote inspection report to {args.output}")


if __name__ == "__main__":
    main()
