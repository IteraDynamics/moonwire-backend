# tests/test_trigger_likelihood_gb.py
from pathlib import Path
import os
from fastapi.testclient import TestClient

from src import paths
from src.ml.train_trigger_model import train
from src.main import app

def test_gb_trains_and_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")
    os.environ["DEMO_MODE"] = "true"
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    out = train(days=1, interval="hour", out_dir=tmp_path / "models")
    # artifacts exist if training succeeded
    assert (tmp_path / "models" / "trigger_likelihood_gb.joblib").exists()

    client = TestClient(app)
    r = client.get("/internal/trigger-likelihood/metadata")
    assert r.status_code == 200
    meta = r.json()
    # nested gb block present
    assert "gb" in meta
    assert "metrics" in meta["gb"]

def test_ensemble_with_gb_vote(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")
    os.environ["DEMO_MODE"] = "true"
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    train(days=1, interval="hour", out_dir=tmp_path / "models")

    client = TestClient(app)
    payload = {"features": {"burst_z": 1.5}}
    r = client.post("/internal/trigger-likelihood/score?use=ensemble", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "prob_trigger_next_6h" in data
    assert "votes" in data
    assert any(k in data["votes"] for k in ("gb","rf","logistic"))
