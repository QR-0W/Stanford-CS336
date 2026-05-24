from pathlib import Path

import fasttext


_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        model_path = Path(__file__).resolve().parent / "assets" / "lid.176.ftz"
        _MODEL = fasttext.load_model(str(model_path))
    return _MODEL


def identify_language(text: str) -> tuple[str, float]:
    text = text.replace("\n", " ").strip()
    model = _get_model()
    pred = model.predict(text, k=1)
    label = pred[0][0].replace("__label__", "")
    score = float(pred[1][0])
    return label, score
