# tests/test_override_suppression.py

import shutil
import json
import pytest
from pathlib import Path

from src.adjustment_trigger_router import get_adaptive_threshold

# helper to seed reviewer_scores.jsonl
def write_scores(scores: dict[str, float]):
    root = Path(__file__).resolve().parent.parent
    logs = root / "logs"
    shutil.rmtree(logs, ignore_errors=True)
    logs.mkdir(parents=True, exist_ok=True)
    scores_path = logs / "reviewer_scores.jsonl"
    with scores_path.open("w") as f:
        for rid, score in scores.items():
            f.write(json.dumps({"reviewer_id": rid, "score": score}) + "\n")


def test_override_low_weight(client):
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
    # low-weight threshold = 0.8 → stays suppressed
    assert data["threshold_used"] == pytest.approx(0.8)
    assert data["unsuppressed"] is False


def test_override_mid_weight(client):
    write_scores({"mid_rev": 0.6})

    payload = {
        "signal_id": "sig_mid",
        "override_reason": "unit-test",
        "reviewer_id": "mid_rev",
        "trust_delta": 0.2,
    }
    r = client.post("/internal/override-suppression", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert data["reviewer_weight"] == pytest.approx(1.0)
    assert data["new_trust_score"] == pytest.approx(0.2)
    # mid-weight threshold = 0.7 → stays suppressed
    assert data["threshold_used"] == pytest.approx(0.7)
    assert data["unsuppressed"] is False


def test_override_high_weight(client):
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
    # high-weight threshold = 0.4 → now unsuppressed
    assert data["threshold_used"] == pytest.approx(0.4)
    assert data["unsuppressed"] is True
