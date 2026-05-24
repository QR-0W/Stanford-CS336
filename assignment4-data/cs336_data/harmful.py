from __future__ import annotations

from pathlib import Path

import fasttext


_NSFW_MODEL = None
_TOXIC_MODEL = None


def _load_model(filename: str):
    model_path = Path(__file__).resolve().parent / "assets" / filename
    return fasttext.load_model(str(model_path))


def _predict(model, text: str) -> tuple[str, float]:
    text = text.replace("\n", " ").strip()
    labels, scores = model.predict(text, k=1)
    return labels[0].replace("__label__", ""), float(scores[0])


def classify_nsfw(text: str) -> tuple[str, float]:
    global _NSFW_MODEL
    if _NSFW_MODEL is None:
        _NSFW_MODEL = _load_model("dolma_fasttext_nsfw_jigsaw_model.bin")
    return _predict(_NSFW_MODEL, text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    global _TOXIC_MODEL
    if _TOXIC_MODEL is None:
        _TOXIC_MODEL = _load_model("dolma_fasttext_hatespeech_jigsaw_model.bin")
    return _predict(_TOXIC_MODEL, text)
