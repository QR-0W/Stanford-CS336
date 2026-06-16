"""语言识别：基于 fastText LID 模型的语言分类。

使用 Meta 官方提供的量化 fastText 语言识别模型
（``lid.176.ftz``，约 917KB），预测文本的 ISO 639-1 语言代码。
模型只加载一次，之后全局复用。

注意：本作业使用 ``.ftz`` 量化版而非 ``.bin`` 原版，
因为原版 ``.bin`` 在当前 fastText 版本下预测异常（固定输出概率 0.25）。
"""

from pathlib import Path

import fasttext

from cs336_data.fasttext_compat import predict_top1


# 全局缓存，避免每次调用都重新加载模型
_MODEL = None


def _get_model():
    """惰性加载 fastText LID 模型。"""
    global _MODEL
    if _MODEL is None:
        model_path = Path(__file__).resolve().parent / "assets" / "lid.176.ftz"
        _MODEL = fasttext.load_model(str(model_path))
    return _MODEL


def identify_language(text: str) -> tuple[str, float]:
    """返回文本的预测语言标签及置信度。

    fastText 输入要求没有换行符，所以先将 ``\\n`` 替换为空格。

    Args:
        text: 待识别的文本片段。

    Returns:
        ``(语言代码, 置信度)`` 元组，如 ``("en", 0.95)``。
    """
    text = text.replace("\n", " ").strip()
    model = _get_model()
    return predict_top1(model, text)
