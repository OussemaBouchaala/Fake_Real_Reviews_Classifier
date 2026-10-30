import numpy as np
import pytest
from fastapi.testclient import TestClient

import api


class FakePipeline:
    """Says REAL (1) unless the review shouts "BUY NOW"."""

    def predict(self, texts):
        return np.array([0 if "BUY NOW" in t else 1 for t in texts])


@pytest.fixture
def client(monkeypatch):
    # Mock the model file: the API's startup calls joblib.load, which now returns the fake pipeline.
    monkeypatch.setattr(api.joblib, "load", lambda path: FakePipeline())
    with TestClient(api.app) as c:  # `with` runs the startup (lifespan) code
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_once_model_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200 and r.json() == {"status": "ready"}


def test_not_ready_before_startup():
    # Without `with`, the lifespan never runs, like a pod that is still loading the model.
    c = TestClient(api.app)
    assert c.get("/ready").status_code == 503
    assert c.get("/health").status_code == 200  # alive, just not ready


@pytest.mark.parametrize(
    "text, label",
    [("Fits well, good fabric.", "REAL"), ("BEST EVER!!! BUY NOW", "FAKE")],
)
def test_predict(client, text, label):
    r = client.post("/predict", json={"text": text})
    assert r.status_code == 200 and r.json() == {"label": label}


@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": "   "}, {"text": 42}])
def test_predict_rejects_bad_input(client, body):
    assert client.post("/predict", json=body).status_code == 422
