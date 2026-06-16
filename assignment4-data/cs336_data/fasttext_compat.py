"""Compatibility helpers for fastText prediction."""

from __future__ import annotations


def predict_top1(model, text: str) -> tuple[str, float]:
    """Return fastText top-1 label and confidence under NumPy 1.x/2.x.

    fastText 0.9.x calls ``np.array(probs, copy=False)`` internally, which
    raises under NumPy 2. The wrapped C++ binding still returns predictions
    correctly, so this helper mirrors ``FastText.predict`` and uses the lower
    level binding directly.
    """
    line = text.replace("\n", " ").strip()
    predictions = model.f.predict(f"{line}\n", 1, 0.0, "strict")
    if not predictions:
        return "", 0.0
    score, label = predictions[0]
    return label.replace("__label__", ""), float(score)
