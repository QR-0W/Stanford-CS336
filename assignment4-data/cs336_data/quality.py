from __future__ import annotations

import re
import math
from pathlib import Path


WORD_RE = re.compile(r"\S+")
BOILERPLATE_RE = re.compile(
    r"\b(?:faq|search|memberlist|usergroups|register|profile|log\s*in|"
    r"private messages|powered by|copyright|forum index)\b",
    re.IGNORECASE,
)
_QUALITY_MODEL = None


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _has_alpha(word: str) -> bool:
    return any(char.isalpha() for char in word)


def gopher_quality_filter(text: str) -> bool:
    words = _words(text)
    num_words = len(words)
    if num_words < 50 or num_words > 100_000:
        return False

    mean_word_length = sum(len(word) for word in words) / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if lines:
        ellipsis_lines = sum(line.endswith("...") for line in lines)
        if ellipsis_lines / len(lines) > 0.30:
            return False

    alpha_word_fraction = sum(_has_alpha(word) for word in words) / num_words
    if alpha_word_fraction < 0.80:
        return False

    return True


def _heuristic_classify_quality(text: str) -> tuple[str, float]:
    words = _words(text)
    if not words:
        return "cc", 1.0

    num_words = len(words)
    alpha_fraction = sum(_has_alpha(word) for word in words) / num_words
    unique_fraction = len({word.lower() for word in words}) / num_words
    boilerplate_hits = len(BOILERPLATE_RE.findall(text))

    if boilerplate_hits >= 4 and num_words < 300:
        return "cc", float(min(0.5 + boilerplate_hits / 10, 1.0))

    score = 0.0
    score += min(math.log1p(num_words) / math.log1p(2_000), 1.0) * 0.35
    score += min(alpha_fraction, 1.0) * 0.25
    score += min(unique_fraction / 0.65, 1.0) * 0.20
    score += (0.20 if gopher_quality_filter(text) else -0.20)
    score -= min(boilerplate_hits / 8, 1.0) * 0.45

    if score >= 0.60:
        return "wiki", float(min(score, 1.0))
    return "cc", float(min(max(1.0 - score, 0.01), 1.0))


def classify_quality(text: str) -> tuple[str, float]:
    global _QUALITY_MODEL
    model_path = Path(__file__).resolve().parent / "assets" / "quality_classifier.bin"
    if model_path.exists():
        if _QUALITY_MODEL is None:
            import fasttext

            _QUALITY_MODEL = fasttext.load_model(str(model_path))
        labels, scores = _QUALITY_MODEL.predict(text.replace("\n", " ").strip(), k=1)
        return labels[0].replace("__label__", ""), float(scores[0])

    return _heuristic_classify_quality(text)
