import json
from pathlib import Path
import importlib
import os

import src.paths as paths
from src.ml.retrain_from_log import retrain_from_log


def _write_training_rows(path: Path):
    rows = [
        {"timestamp": "2025-01-01T00:00:00Z", "origin": "reddit",  "features": {"burst_z": 1.0}, "label": True},
        {"timestamp": "2025-01-01T01:00:00Z", "origin": "reddit",  "features": {"burst_z": 0.2}, "label": False},
        {"timestamp": "2025-01-01T02:00:00Z", "origin": "twitter", "features": {"burst_z": 2.0}, "label": True},
        {"timestamp": "2025-01-01T03:00:00Z", "origin": "twitter", "features": {"burst_z": 0.1}, "label": False},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_retrain_writes_version_and_infer_logs(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    monkeypatch.setattr(paths, "MODELS_DIR", model_dir)
    td_path = model_dir / "training_data.jsonl"
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_training_rows(td_path)

    retrain_from_log(train_log_path=td_path, save_dir=model_dir, version="v-test")

    ver_file = model_dir / "training_version.txt"
    assert ver_file.exists()
    assert ver_file.read_text().strip() == "v-test"

    trig_log = model_dir / "trigger_history.jsonl"
    monkeypatch.setenv("TRIGGER_LOG_PATH", str(trig_log))
    import src.ml.infer as infer
    importlib.reload(infer)

    payload = {"origin": "reddit", "features": {"burst_z": 1.5}}
    infer.infer_score_ensemble(payload, models_dir=model_dir)

    lines = trig_log.read_text().splitlines()
    assert lines
    last = json.loads(lines[-1])
    assert last.get("model_version") == "v-test"

