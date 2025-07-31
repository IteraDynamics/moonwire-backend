# tests/test_reviewer_impact_log.py

import pytest


def test_reviewer_impact_log_and_scores(client):
    payload = {
        "signal_id": "test123",
        "reviewer_id": "alice",
        "action": "override",
        "trust_delta": 0.1,
    }

    # 1) Log the action
    r1 = client.post("/internal/reviewer-impact-log", json=payload)
    assert r1.status_code == 200

    data1 = r1.json()
    # status key unchanged
    assert data1.get("status") == "logged"
    # now also returns reviewer_weight
    assert "reviewer_weight" in data1
    # initial reviewer score = 0 → weight should be 0.75
    assert isinstance(data1["reviewer_weight"], (int, float))
    assert data1["reviewer_weight"] == 0.75

    # 2) Trigger a recompute of reviewer scores
    r2 = client.post("/internal/trigger-reviewer-scoring")
    assert r2.status_code == 200
    assert r2.json().get("recomputed") is True

    # 3) Fetch the updated scores
    r3 = client.get("/internal/reviewer-scores")
    assert r3.status_code == 200
    scores = r3.json().get("scores", [])
    # alice took exactly 1 action, so her score should now be 1
    assert any(item["reviewer_id"] == "alice" and item["score"] == 1 for item in scores)