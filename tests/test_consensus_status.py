# tests/test_consensus_status.py

import json
import time
from pathlib import Path
import pytest

from src.paths import (
    RETRAINING_LOG_PATH,
    REVIEWER_SCORES_PATH,
    RETRAINING_TRIGGERED_LOG_PATH,
    LOGS_DIR
)

def ensure_logs_exist():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RETRAINING_LOG_PATH.touch(exist_ok=True)
    REVIEWER_SCORES_PATH.touch(exist_ok=True)
    RETRAINING_TRIGGERED_LOG_PATH.touch(exist_ok=True)

def append_jsonl(path: Path, obj: dict):
    ensure_logs_exist()
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")

def write_flag(signal_id: str, reviewer_id: str, weight: float = None):
    entry = {
        "signal_id": signal_id,
        "reviewer_id": reviewer_id,
        "timestamp": time.time()
    }
    if weight is not None:
        entry["reviewer_weight"] = weight
    append_jsonl(RETRAINING_LOG_PATH, entry)

def write_score(reviewer_id: str, score: float):
    append_jsonl(REVIEWER_SCORES_PATH, {"reviewer_id": reviewer_id, "score": score})

def read_trigger_log():
    ensure_logs_exist()
    if RETRAINING_TRIGGERED_LOG_PATH.exists():
        with RETRAINING_TRIGGERED_LOG_PATH.open("r") as f:
            return [json.loads(line) for line in f if line.strip()]
    return []

def clear_logs():
    ensure_logs_exist()
    RETRAINING_LOG_PATH.write_text("")
    REVIEWER_SCORES_PATH.write_text("")
    RETRAINING_TRIGGERED_LOG_PATH.write_text("")

def test_trigger_not_met(client):
    clear_logs()
    write_flag("sig-low", "r1", weight=1.0)
    r = client.post("/internal/evaluate-consensus-retraining", json={"signal_id": "sig-low"})
    assert r.status_code == 200
    assert r.json()["triggered"] is False
    assert r.json()["total_weight"] == 1.0

def test_trigger_met(client):
    clear_logs()
    write_flag("sig-high", "r1", weight=1.0)
    write_flag("sig-high", "r2", weight=1.25)
    write_flag("sig-high", "r3", weight=0.75)
    r = client.post("/internal/evaluate-consensus-retraining", json={"signal_id": "sig-high"})
    assert r.status_code == 200
    assert r.json()["triggered"] is True
    assert r.json()["total_weight"] == pytest.approx(3.0)

def test_mixed_scores_and_weights(client):
    clear_logs()
    write_score("rA", 0.82)  # → 1.25
    write_score("rB", 0.60)  # → 1.0
    write_score("rC", 0.40)  # → 0.75
    write_flag("sig-mixed", "rA")
    write_flag("sig-mixed", "rB")
    write_flag("sig-mixed", "rC")
    r = client.post("/internal/evaluate-consensus-retraining", json={"signal_id": "sig-mixed"})
    assert r.status_code == 200
    assert r.json()["triggered"] is True
    assert r.json()["total_weight"] == pytest.approx(3.0)

def test_no_reviewers_returns_triggered_false(client):
    clear_logs()
    r = client.post("/internal/evaluate-consensus-retraining", json={"signal_id": "sig-none"})
    assert r.status_code == 200
    assert r.json()["triggered"] is False
    assert r.json()["total_weight"] == 0.0
    assert r.json()["reviewers"] == []

def test_log_written_on_trigger(client):
    clear_logs()
    write_flag("sig-log", "r1", weight=2.6)
    r = client.post("/internal/evaluate-consensus-retraining", json={"signal_id": "sig-log"})
    assert r.status_code == 200
    assert r.json()["triggered"] is True
    logs = read_trigger_log()
    found = any(entry["signal_id"] == "sig-log" for entry in logs)
    assert found
