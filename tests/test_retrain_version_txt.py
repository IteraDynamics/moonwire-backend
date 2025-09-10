import json
from pathlib import Path

from src.ml import retrain_from_log, training_metadata


def _make_rows():
    rows = []
    for i in range(4):
        rows.append({
            "origin": "reddit",
            "features": {"f1": float(i)},
            "label": bool(i % 2),
        })
    return rows


def test_retrain_writes_version_file(tmp_path, monkeypatch):
    train_log = tmp_path / "training_data.jsonl"
    train_log.write_text("\n".join(json.dumps(r) for r in _make_rows()) + "\n")

    # Ensure retrain_from_log and training_metadata use tmp_path as MODELS_DIR
    monkeypatch.setattr(retrain_from_log, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(training_metadata, "RUNS_PATH", tmp_path / "training_runs.jsonl")

    retrain_from_log.retrain_from_log(train_log_path=train_log, save_dir=tmp_path, version="vTEST")

    assert (tmp_path / "training_version.txt").read_text().strip() == "vTEST"
    latest = training_metadata.load_latest_training_metadata(runs_path=tmp_path / "training_runs.jsonl", allow_demo_seed=False)
    assert latest["version"] == "vTEST"
