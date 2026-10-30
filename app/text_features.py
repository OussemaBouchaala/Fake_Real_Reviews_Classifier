"""Text helpers used inside the scikit-learn Pipeline.

They live in a module (not in the notebook) so the saved pipeline can be
loaded by the Streamlit app and the API: joblib stores functions by import path.
"""
import re
import string
from typing import Any, Iterable, Literal, Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_PUNCT = str.maketrans("", "", string.punctuation)

Label = Literal["REAL", "FAKE"]


class Classifier(Protocol):
    """Anything with a scikit-learn style predict(): the trained pipeline, or a fake one in tests."""

    def predict(self, X: Sequence[str]) -> Any: ...


def clean_text(text: object) -> str:
    """Lowercase, drop punctuation, digits and English stop words."""
    text = str(text).lower().translate(_PUNCT)
    text = re.sub(r"\d+", "", text)
    return " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)


def clean_texts(texts: Iterable[object]) -> list[str]:
    return [clean_text(t) for t in texts]


def behavior_features(texts: Iterable[object]) -> np.ndarray:
    """Per-review style signals: length, word count, '!', capitals, repetition, mean word length."""
    rows: list[list[float]] = []
    for raw in texts:
        t = str(raw)
        words = t.split()
        n = max(len(words), 1)
        rows.append([
            len(t),
            len(words),
            t.count("!"),
            sum(c.isupper() for c in t) / max(len(t), 1),
            1 - len(set(w.lower() for w in words)) / n,
            sum(len(w) for w in words) / n,
        ])
    return np.array(rows, dtype=float)


def to_frame(texts: Iterable[object]) -> pd.DataFrame:
    """Wrap raw review texts in the one-column DataFrame the pipeline expects."""
    return pd.DataFrame({"text": [str(t) for t in texts]})


def predict_label(model: Classifier, text: str) -> Label:
    """Classify one review with the trained pipeline: "REAL" (label 1) or "FAKE" (label 0)."""
    return "REAL" if model.predict([text])[0] == 1 else "FAKE"
