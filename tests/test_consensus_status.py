# tests/test_consensus_status.py

import json
import pytest

def read_trigger_log():
    # Resolve the path dynamically so it respects per-test LOGS_DIR monkeypatching
    import importlib
    import src.paths
    importlib.reload(src.paths)
    from src.paths import RETRAINING_TRIGGERED_LOG_PATH

    if RETRAINING_TRIGGERED_LOG_PATH.exists():
        with RETRAINING_TRIGGERED_LOG_PATH.open("r") as f:
            return [json.loads(line) for line in f if line.strip()]
    return []

def test_trigger_not_met(client, write_flag):
    write_flag("sig-low", "r1", weight=1.0)
    r = client.post("/internal/evaluate-consensus-retraining", json={"signal_id": "sig-low"})
    assert r.status_code == 200
    assert r.json()["triggered"] is False
    assert r.json()["total_weight"] == pytest.approx(1.0)
    assert len(r.json()["reviewers"]) == 1

def test_trigger_met(client, write_flag):
    write_flag("sig-high", "r1", weight=1.0)
    write_flag("sig-high", "r2", weight=1.25)
    write_flag("sig-high", "r3", weight=0.75)
    r = client.post("/internal/evaluate-consensus-retraining", json={"signal_id": "sig-high"})
    assert r.status_code == 200
    assert r.json()["triggered"] is True
    assert r.json()["total_weight"] == pytest.approx(3.0)
    assert len(r.json()["reviewers"]) == 3

def test_mixed_scores_and_weights(client, write_flag, write_score):
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
    assert len(r.json()["reviewers"]) == 3

def test_no_reviewers_returns_triggered_false(client):
    r = client.post("/internal/evaluate-consensus-retraining", json={"signal_id": "sig-none"})
    assert r.status_code == 200
    assert r.json()["triggered"] is False
    assert r.json()["total_weight"] == 0.0
    assert r.json()["reviewers"] == []

def test_log_written_on_trigger(client, write_flag):
    write_flag("sig-log", "r1", weight=2.6)
    r = client.post("/internal/evaluate-consensus-retraining", json={"signal_id": "sig-log"})
    assert r.status_code == 200
    assert r.json()["triggered"] is True

    logs = read_trigger_log()
    assert any(entry["signal_id"] == "sig-log" for entry in logs)