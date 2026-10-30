import numpy as np
import pandas as pd
import pytest

from text_features import behavior_features, clean_text, clean_texts, predict_label, to_frame


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Great PRODUCT", "great product"),                    # lowercased
        ("Love it!!! Works, really.", "love works really"),    # punctuation and stop words ("it") removed
        ("Bought 3 of them in 2024", "bought"),                # digits removed, "of them in" are stop words
        ("   spaced    out   ", "spaced"),                     # extra whitespace collapsed, "out" is a stop word
        ("", ""),                                              # empty review stays empty
        (12345, ""),                                           # non-strings are converted, not crashed on
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_texts_cleans_each_item():
    assert clean_texts(["Nice!", "The best"]) == ["nice", "best"]


@pytest.mark.parametrize(
    "text, column, expected",
    [
        ("abc def", 0, 7),              # length in characters
        ("one two three", 1, 3),        # word count
        ("wow!! great!", 2, 3),         # exclamation marks
        ("ABcd", 3, 0.5),               # share of capital letters
        ("good good good bad", 4, 0.5), # repetition: 1 - 2 unique / 4 words
        ("ab abcd", 5, 3.0),            # mean word length
    ],
)
def test_behavior_features_columns(text, column, expected):
    assert behavior_features([text])[0, column] == pytest.approx(expected)


def test_behavior_features_empty_review_has_no_division_by_zero():
    row = behavior_features([""])[0]
    assert row.tolist() == [0, 0, 0, 0, 1, 0]
    assert np.isfinite(row).all()


def test_behavior_features_shape():
    assert behavior_features(["a", "b c", "d e f"]).shape == (3, 6)


def test_to_frame():
    df = to_frame(["ok", 5])
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == ["text"]
    assert df["text"].tolist() == ["ok", "5"]


class FakeModel:
    """Stands in for the trained pipeline: records what it was given and returns a fixed label."""

    def __init__(self, label):
        self.label, self.seen = label, None

    def predict(self, texts):
        self.seen = texts
        return np.array([self.label])


@pytest.mark.parametrize("label, expected", [(1, "REAL"), (0, "FAKE")])
def test_predict_label(label, expected):
    model = FakeModel(label)
    assert predict_label(model, "Nice shoes!") == expected
    assert model.seen == ["Nice shoes!"]  # raw text in a list: the pipeline does its own cleaning
