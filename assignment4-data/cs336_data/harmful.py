"""有害内容检测：NSFW 和毒性言论分类。

使用 Dolma 项目提供的 fastText 模型，基于 Jigsaw 评论数据训练：
    - ``dolma_fasttext_nsfw_jigsaw_model.bin``：NSFW 分类
    - ``dolma_fasttext_hatespeech_jigsaw_model.bin``：毒性/仇恨言论分类

两个模型均只加载一次，全局缓存复用。

限制：这两个模型在 Jigsaw 英文评论上训练，
对中文/繁体成人站模板和非英语网页的召回率很低，不宜作为唯一过滤依据。
"""

from __future__ import annotations

from pathlib import Path

import fasttext


# 全局缓存，避免重复加载 .bin 模型文件
_NSFW_MODEL = None
_TOXIC_MODEL = None


def _load_model(filename: str):
    """从 ``cs336_data/assets/`` 目录加载 fastText 模型。"""
    model_path = Path(__file__).resolve().parent / "assets" / filename
    return fasttext.load_model(str(model_path))


def _predict(model, text: str) -> tuple[str, float]:
    """对标准化后的文本执行 fastText top-1 预测。

    将换行符替换为空格以满足 fastText 输入要求，
    并去掉返回标签中的 ``__label__`` 前缀。
    """
    text = text.replace("\n", " ").strip()
    labels, scores = model.predict(text, k=1)
    return labels[0].replace("__label__", ""), float(scores[0])


def classify_nsfw(text: str) -> tuple[str, float]:
    """判断文本是否包含 NSFW 内容。

    Returns:
        ``("nsfw" | "non-nsfw", 置信度)``。
    """
    global _NSFW_MODEL
    if _NSFW_MODEL is None:
        _NSFW_MODEL = _load_model("dolma_fasttext_nsfw_jigsaw_model.bin")
    return _predict(_NSFW_MODEL, text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """判断文本是否包含毒性/仇恨言论。

    Returns:
        ``("toxic" | "non-toxic", 置信度)``。
    """
    global _TOXIC_MODEL
    if _TOXIC_MODEL is None:
        _TOXIC_MODEL = _load_model("dolma_fasttext_hatespeech_jigsaw_model.bin")
    return _predict(_TOXIC_MODEL, text)
