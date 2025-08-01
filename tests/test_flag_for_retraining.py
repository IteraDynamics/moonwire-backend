import os
import json
import shutil
import pytest
from pathlib import Path


# helper to seed reviewer_scores.jsonl
def write_scores(scores: dict[str, float]):
    root = Path(__file__).resolve().parent.parent
    logs = root / "logs"
    shutil.rmtree(logs, ignore_errors=True)
    logs.mkdir(parents=True, exist_ok=True)
    scores_path = logs / "reviewer_scores.jsonl"
    with scores_path.open("w") as f:
        for reviewer_id, score in scores.items():
            f.write(json.dumps({"reviewer_id": reviewer_id, "score": score}) + "\n")


# helper to read retraining_log.jsonl
def read_retrain_log():
    root = Path(__file__).resolve().parent.parent
    log_path = root / "logs" / "retraining_log.jsonl"
    with log_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def test_flag_fallback_weight(client):
    """
    If no reviewer_scores.jsonl exists, default weight=1.0
    """
    # ensure no scores file
    root = Path(__file__).resolve().parent.parent
    shutil.rmtree(root / "logs", ignore_errors=True)

    payload = {
        "signal_id":   "test_fallback",
        "reviewer_id": "unknown_rev",
        "reason":      "unit-test",
        "note":        "fallback case"
    }
    r = client.post("/internal/flag-for-retraining", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert data["reviewer_weight"] == pytest.approx(1.0)
    assert data["status"] == "queued"

    entries = read_retrain_log()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["reviewer_weight"] == pytest.approx(1.0)
    assert entry["signal_id"] == "test_fallback"


def test_flag_known_weight(client):
    """
    If reviewer_score file has an entry, weight is computed correctly.
    """
    write_scores({"alice": 0.8, "bob": 0.3})

    # alice -> score 0.8 => weight 1.25
    payload = {
        "signal_id":   "sig1",
        "reviewer_id": "alice",
        "reason":      "unit-test-known",
    }
    r = client.post("/internal/flag-for-retraining", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert data["reviewer_weight"] == pytest.approx(1.25)

    entries = read_retrain_log()
    # last entry should correspond
    last = entries[-1]
    assert last["reviewer_id"] == "alice"
    assert last["reviewer_weight"] == pytest.approx(1.25)
    assert last["signal_id"] == "sig1"
