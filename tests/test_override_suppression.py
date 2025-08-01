# tests/test_override_suppression.py

import os
import json
import shutil
import pytest
from pathlib import Path

# helper to write a reviewer_scores.jsonl in the real logs folder
def write_scores(reviewer_scores: dict[str, float]):
    repo_root = Path(__file__).resolve().parent.parent
    logs_dir = repo_root / "logs"
    # clear & recreate
    shutil.rmtree(logs_dir, ignore_errors=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    scores_file = logs_dir / "reviewer_scores.jsonl"
    with scores_file.open("w") as f:
        for reviewer_id, score in reviewer_scores.items():
            f.write(json.dumps({"reviewer_id": reviewer_id, "score": score}) + "\n")


def test_override_low_weight(client):
    """
    A low-score reviewer (<0.5) has weight=0.75,
    so with trust_delta=0.1 new_score=0.075 which < 0.4 → unsuppressed=False
    """
    write_scores({"low_rev": 0.0})

    payload = {
        "signal_id": "sig_low",
        "override_reason": "unit-test",
        "reviewer_id": "low_rev",
        "trust_delta": 0.1,
    }
    r = client.post("/internal/override-suppression", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert data["reviewer_weight"] == pytest.approx(0.75)
    assert data["new_trust_score"] == pytest.approx(0.075)
    assert data["unsuppressed"] is False


def test_override_high_weight(client):
    """
    A high-score reviewer (>=0.75) has weight=1.25,
    so with trust_delta=0.4 new_score=0.5 which >= 0.4 → unsuppressed=True
    """
    write_scores({"high_rev": 1.0})

    payload = {
        "signal_id": "sig_high",
        "override_reason": "unit-test",
        "reviewer_id": "high_rev",
        "trust_delta": 0.4,
    }
    r = client.post("/internal/override-suppression", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert data["reviewer_weight"] == pytest.approx(1.25)
    assert data["new_trust_score"] == pytest.approx(0.4 * 1.25)
    assert data["unsuppressed"] is True