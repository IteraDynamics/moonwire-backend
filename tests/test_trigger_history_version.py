# tests/test_trigger_history_version.py
from pathlib import Path
import json
import importlib

def test_trigger_history_contains_model_version(tmp_path: Path, monkeypatch):
    # Point trigger history to a temp file BEFORE importing the module
    log_path = tmp_path / "trigger_history.jsonl"
    monkeypatch.setenv("TRIGGER_LOG_PATH", str(log_path))

    # Import (or reload) infer so it picks up env at import time
    import src.ml.infer as infer
    importlib.reload(infer)  # ensure it re-reads env & module-level constants

    # Create a temp version file where infer reads it from
    from src.paths import MODELS_DIR
    (MODELS_DIR / "training_version.txt").write_text("v.test\n")

    # Run one inference to produce a history entry
    _ = infer.infer_score_ensemble({"origin": "reddit", "features": {"burst_z": 1.0}})

    # Read the last history line and assert model_version is present
    lines = log_path.read_text().strip().splitlines()
    assert lines, "no history lines written"
    last = json.loads(lines[-1])
    assert last.get("model_version") == "v.test", last
    assert last.get("origin") == "reddit", last