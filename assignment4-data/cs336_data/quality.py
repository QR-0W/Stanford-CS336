"""文档质量过滤：Gopher 规则 + wiki-vs-CC 分类器。

Gopher 规则:
    ``gopher_quality_filter``
        四条可解释规则（词数、平均词长、省略号行比例、字母词比例）。

质量分类器:
    ``classify_quality``
        优先使用 ``cs336_data/assets/quality_classifier.bin`` 中的
        fastText 二分类模型（以 Wikipedia reference pages 为正例、
        Common Crawl 为负例训练）。若模型文件不存在，回退到启发式打分器。
"""

from __future__ import annotations

import re
import math
from pathlib import Path


# 用 whitespace 分词（不依赖 NLTK，避免额外资源下载）
WORD_RE = re.compile(r"\S+")

# 锅炉房/论坛模板关键词，用于启发式分类器识别低质量模板页面
BOILERPLATE_RE = re.compile(
    r"\b(?:faq|search|memberlist|usergroups|register|profile|log\s*in|"
    r"private messages|powered by|copyright|forum index)\b",
    re.IGNORECASE,
)

# fastText 质量分类器全局缓存
_QUALITY_MODEL = None


def _words(text: str) -> list[str]:
    """按连续非空白字符分词（whitespace tokenization）。"""
    return WORD_RE.findall(text)


def _has_alpha(word: str) -> bool:
    """判断 token 中是否至少包含一个字母字符。"""
    return any(char.isalpha() for char in word)


def gopher_quality_filter(text: str) -> bool:
    """Gopher 论文子集：四条规则全部通过才返回 ``True``。

    规则：
        1. 词数在 [50, 100,000] 之间
        2. 平均词长在 [3, 10] 之间
        3. 不超过 30% 的非空行以 ``...`` 结尾
        4. 至少 80% 的 token 含字母字符
    """
    words = _words(text)
    num_words = len(words)
    if num_words < 50 or num_words > 100_000:
        return False

    mean_word_length = sum(len(word) for word in words) / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # 省略号行比例：很多 copy-paste 页面或目录页以 "..." 结尾
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if lines:
        ellipsis_lines = sum(line.endswith("...") for line in lines)
        if ellipsis_lines / len(lines) > 0.30:
            return False

    # 含字母词比例：过低说明大部分是数字/符号（乱码或二进制内容）
    alpha_word_fraction = sum(_has_alpha(word) for word in words) / num_words
    if alpha_word_fraction < 0.80:
        return False

    return True


def _heuristic_classify_quality(text: str) -> tuple[str, float]:
    """启发式质量打分（fastText 模型不可用时的回退方案）。

    打分维度：文本长度、字母比例、词汇多样性、Gopher 结果、锅炉房关键词命中。
    综合分数 >= 0.60 判为 ``"wiki"``，否则 ``"cc"``。
    """
    words = _words(text)
    if not words:
        return "cc", 1.0

    num_words = len(words)
    alpha_fraction = sum(_has_alpha(word) for word in words) / num_words
    unique_fraction = len({word.lower() for word in words}) / num_words
    boilerplate_hits = len(BOILERPLATE_RE.findall(text))

    # 锅炉房关键词密集且文本短 -> 几乎确定是模板/论坛页
    if boilerplate_hits >= 4 and num_words < 300:
        return "cc", float(min(0.5 + boilerplate_hits / 10, 1.0))

    # 综合打分：各维度加权求和
    score = 0.0
    # 文本长度（对数尺度归一化到 [0,1]）
    score += min(math.log1p(num_words) / math.log1p(2_000), 1.0) * 0.35
    # 字母比例
    score += min(alpha_fraction, 1.0) * 0.25
    # 词汇多样性（唯一词比例）
    score += min(unique_fraction / 0.65, 1.0) * 0.20
    # Gopher 规则作为额外特征
    score += (0.20 if gopher_quality_filter(text) else -0.20)
    # 锅炉房惩罚
    score -= min(boilerplate_hits / 8, 1.0) * 0.45

    if score >= 0.60:
        return "wiki", float(min(score, 1.0))
    return "cc", float(min(max(1.0 - score, 0.01), 1.0))


def classify_quality(text: str) -> tuple[str, float]:
    """质量分类：优先使用 fastText 模型，否则回退到启发式。

    Args:
        text: 待分类的文本。

    Returns:
        ``(label, confidence)``，其中 label 为 ``"wiki"`` 或 ``"cc"``。
    """
    global _QUALITY_MODEL
    model_path = Path(__file__).resolve().parent / "assets" / "quality_classifier.bin"
    if model_path.exists():
        if _QUALITY_MODEL is None:
            import fasttext

            _QUALITY_MODEL = fasttext.load_model(str(model_path))
        # fastText 输入不能含换行符，需先标准化
        labels, scores = _QUALITY_MODEL.predict(text.replace("\n", " ").strip(), k=1)
        return labels[0].replace("__label__", ""), float(scores[0])

    return _heuristic_classify_quality(text)
