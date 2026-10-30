import io
import json
from pathlib import Path

import utils

FAKE = {"paths": {k: f"../fake/{k}" for k in [
    "models_dir", "data_dir", "notebooks_dir", "app_dir", "tfidf_vectorizer", "svm_model",
    "svm_model_basic", "review_pipeline", "fake_reviews_dataset", "teams_dataset", "notebook"]}}


def test_load_config_reads_config_json_next_to_app(monkeypatch):
    opened = []

    def fake_open(path, mode="r", *a, **kw):
        opened.append((Path(path), mode))
        return io.StringIO(json.dumps(FAKE))

    # A module-level `open` in utils shadows the builtin, so only load_config() sees the fake.
    monkeypatch.setattr(utils, "open", fake_open, raising=False)

    assert utils.load_config() == FAKE
    assert opened == [(Path(utils.__file__).parent.parent / "config.json", "r")]


def test_config_module_exposes_paths(monkeypatch):
    monkeypatch.setattr(utils, "load_config", lambda: FAKE)
    import importlib, sys
    sys.modules.pop("config", None)
    config = importlib.import_module("config")
    assert config.SVM_MODEL == "../fake/svm_model"
    assert config.REVIEW_PIPELINE == "../fake/review_pipeline"
    assert config.FAKE_REVIEWS_DATASET == "../fake/fake_reviews_dataset"
    sys.modules.pop("config", None)  # don't leak the fake config into other tests


def test_real_config_json_is_valid():
    paths = utils.load_config()["paths"]
    assert set(FAKE["paths"]) <= set(paths)
