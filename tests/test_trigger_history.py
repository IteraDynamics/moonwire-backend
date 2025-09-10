import json
from src.ml import infer


def _read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_logistic_trigger_history(tmp_path):
    log_file = tmp_path / "log.jsonl"
    orig = infer._TRIGGER_LOG_PATH
    infer._TRIGGER_LOG_PATH = log_file
    try:
        payload = {"origin": "test", "features": {"burst_z": 1.0}, "threshold": 0.5}
        infer.infer_score(payload)
        records = _read_log(log_file)
        assert len(records) == 1
        rec = records[0]
        for key in ("model_version", "threshold", "decision", "features"):
            assert key in rec
        assert rec["features"] == payload["features"]
    finally:
        infer._TRIGGER_LOG_PATH = orig
        if log_file.exists():
            log_file.unlink()


def test_ensemble_trigger_history(tmp_path):
    log_file = tmp_path / "log.jsonl"
    orig = infer._TRIGGER_LOG_PATH
    infer._TRIGGER_LOG_PATH = log_file
    try:
        payload = {"origin": "test", "features": {"burst_z": 1.0}, "threshold": 0.5}
        infer.infer_score_ensemble(payload)
        records = _read_log(log_file)
        assert len(records) == 1
        rec = records[0]
        for key in ("model_version", "threshold", "decision", "features"):
            assert key in rec
        assert rec["features"] == payload["features"]
    finally:
        infer._TRIGGER_LOG_PATH = orig
        if log_file.exists():
            log_file.unlink()
