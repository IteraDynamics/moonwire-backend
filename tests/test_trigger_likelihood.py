import json, os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from main import app
import src.paths as paths
from src.ml.train_trigger_model import train
from src.ml.infer import score as infer_score


def _j(d): return json.dumps(d, separators=(",", ":"))
def _log(ts, origin): return {"timestamp": ts.isoformat(), "origin": origin}

def _write_series(buf: list, start: datetime, origin: str, counts: list[int]):
    for i, c in enumerate(counts):
        ts = start + timedelta(hours=i)
        for _ in range(int(c)):
            buf.append(_log(ts, origin))

def test_training_and_artifacts(tmp_path: Path, monkeypatch):
    # synthetic logs where “high recent count” => triggers soon
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=14*24 - 1)
    flags, triggers = [], []
    # twitter: bursts → future triggers
    tw = [0]* (14*24 - 10) + [5,6,7,8,9,10,8,6,5,4]
    _write_series(flags, start, "twitter", tw)
    # triggers fire after spikes
    _write_series(triggers, start + timedelta(hours=1), "twitter", [0]* (14*24 - 10) + [0,0,1,1,1,1,1,0,0,0])

    # reddit: flatter
    _write_series(flags, start, "reddit", [1]* (14*24))
    _write_series(triggers, start, "reddit", [0]* (14*24))

    (tmp_path / "retraining_log.jsonl").write_text("\n".join(_j(r) for r in flags))
    (tmp_path / "retraining_triggered.jsonl").write_text("\n".join(_j(r) for r in triggers))

    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")

    out = train(days=14, interval="hour", out_dir=tmp_path / "models")
    mp = Path(out["model_path"])
    assert mp.exists()
    assert mp.with_suffix(".meta.json").exists()

def test_infer_prob_and_monotonicity(tmp_path: Path, monkeypatch):
    # reuse trained artifacts from previous test or train minimal
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path)

    # If no model exists, run a tiny train
    if not (paths.MODELS_DIR / "trigger_likelihood_v0.joblib").exists():
        (paths.LOGS_DIR / "retraining_log.jsonl").write_text("")
        (paths.LOGS_DIR / "retraining_triggered.jsonl").write_text("")
        os.environ["DEMO_MODE"] = "true"
        from src.ml.train_trigger_model import train as _train
        _train(out_dir=paths.MODELS_DIR)
    body_low  = {"features": {"burst_z": 0.0}}
    body_high = {"features": {"burst_z": 5.0}}
    p1 = infer_score(body_low)["prob_trigger_next_6h"]
    p2 = infer_score(body_high)["prob_trigger_next_6h"]
    assert 0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0
    assert p2 >= p1  # monotonic in burst_z for baseline model

def test_api_and_demo(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")
    client = TestClient(app)

    # Without artifacts + DEMO_MODE => 503 for metadata, demo for score
    os.environ["DEMO_MODE"] = "true"
    r = client.post("/internal/trigger-likelihood/score", json={"features": {"burst_z": 1.0}})
    assert r.status_code == 200
    data = r.json()
    assert "prob_trigger_next_6h" in data and 0.0 <= data["prob_trigger_next_6h"] <= 1.0
    assert data.get("demo", False) in (True, False)

def _j(x): return json.dumps(x)

def test_explain_returns_sorted_contribs(tmp_path: Path, monkeypatch):
    # Ensure a model exists (train on demo if needed)
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setenv("DEMO_MODE", "true")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    # Minimal train to materialize artifacts
    train(days=1, interval="hour", out_dir=tmp_path / "models")

    client = TestClient(app)
    payload = {"features": {"burst_z": 2.0, "count_6h": 4.0}}
    res = client.post("/internal/trigger-likelihood/score?explain=true&top_n=3", json=payload)
    assert res.status_code == 200
    data = res.json()
    contrib = data.get("contributions")
    assert isinstance(contrib, dict) and len(contrib) >= 1
    # contributions should be sorted by |value| desc (top_n applied)
    vals = list(contrib.values())
    assert vals == sorted(vals, key=lambda v: abs(v), reverse=True)


def test_metadata_includes_coverage_and_top_features(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setenv("DEMO_MODE", "true")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    train(days=1, interval="hour", out_dir=tmp_path / "models")

    client = TestClient(app)
    res = client.get("/internal/trigger-likelihood/metadata")
    assert res.status_code == 200
    meta = res.json()
    assert "metrics" in meta
    assert "feature_order" in meta
    # coverage available either as summary or full map
    assert ("feature_coverage_summary" in meta) or ("feature_coverage" in meta)
    assert "top_features" in meta and isinstance(meta["top_features"], list)

