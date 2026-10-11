"""Text helpers used inside the scikit-learn Pipeline.

They live in a module (not in the notebook) so the saved pipeline can be
loaded by the Streamlit app: joblib stores functions by import path.
"""
import re
import string

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_PUNCT = str.maketrans("", "", string.punctuation)


def clean_text(text):
    """Lowercase, drop punctuation, digits and English stop words."""
    text = str(text).lower().translate(_PUNCT)
    text = re.sub(r"\d+", "", text)
    return " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)


def clean_texts(texts):
    return [clean_text(t) for t in texts]


def behavior_features(texts):
    """Per-review style signals: length, word count, '!', capitals, repetition, mean word length."""
    rows = []
    for t in texts:
        t = str(t)
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


def to_frame(texts):
    """Wrap raw review texts in the one-column DataFrame the pipeline expects."""
    import pandas as pd
    return pd.DataFrame({"text": [str(t) for t in texts]})
