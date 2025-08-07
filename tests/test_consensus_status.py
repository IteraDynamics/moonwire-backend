# tests/test_consensus_status.py

import json
import time
from pathlib import Path
import pytest

from src.paths import RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH

def append_jsonl(path: Path, obj: dict):
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

def test_single_reviewer(client):
    write_flag("sig-123", "r1", weight=1.1)
    res = client.get("/internal/consensus-status/sig-123")
    assert res.status_code == 200
    assert res.json()["combined_weight"] == 1.1
    assert res.json()["total_reviewers"] == 1

def test_multiple_reviewers(client):
    write_flag("sig-abc", "a", weight=1.0)
    write_flag("sig-abc", "b", weight=1.25)
    write_flag("sig-abc", "c", weight=0.75)
    res = client.get("/internal/consensus-status/sig-abc")
    assert res.status_code == 200
    assert res.json()["total_reviewers"] == 3
    assert res.json()["combined_weight"] == pytest.approx(3.0)

def test_duplicate_reviewers_not_counted_twice(client):
    write_flag("sig-dedupe", "r1", weight=1.1)
    write_flag("sig-dedupe", "r1", weight=0.9)
    res = client.get("/internal/consensus-status/sig-dedupe")
    assert res.status_code == 200
    assert res.json()["total_reviewers"] == 1
    assert res.json()["combined_weight"] == 1.1

def test_missing_weight_with_score_fallback(client):
    write_score("r2", 0.82)  # → 1.25
    write_flag("sig-fallback", "r2")
    res = client.get("/internal/consensus-status/sig-fallback")
    assert res.status_code == 200
    assert res.json()["combined_weight"] == 1.25

def test_missing_weight_and_score_defaults_to_1(client):
    write_flag("sig-default", "ghost")  # no score
    res = client.get("/internal/consensus-status/sig-default")
    assert res.status_code == 200
    assert res.json()["combined_weight"] == 1.0

def test_not_found(client):
    res = client.get("/internal/consensus-status/not-real")
    assert res.status_code == 404