# tests/test_trigger_history_version.py
from pathlib import Path
import json
from src.ml.infer import infer_score_ensemble

def test_trigger_history_contains_model_version(tmp_path: Path, monkeypatch):
    # point logs to a temp file
    monkeypatch.setenv("TRIGGER_LOG_PATH", str(tmp_path / "trigger_history.jsonl"))
    # stub a version file
    (tmp_path / "training_version.txt").write_text("v.test\n")
    # run one inference that triggers logging
    out = infer_score_ensemble({"origin": "reddit", "features": {"burst_z": 1.0}})
    # read last line
    hist = (tmp_path / "trigger_history.jsonl").read_text().strip().splitlines()
    assert hist, "no history lines written"
    last = json.loads(hist[-1])
    assert last.get("model_version") == "v.test"