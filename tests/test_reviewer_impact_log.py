def test_reviewer_impact_log_and_scores(client):
    payload = {
        "signal_id":   "test123",
        "reviewer_id": "alice",
        "action":      "override",
        "trust_delta": 0.1,
    }

    # 1) Log the action
    r1 = client.post("/internal/reviewer-impact-log", json=payload)
    assert r1.status_code == 200
    assert r1.json() == {"status": "logged"}

    # 2) Trigger scoring
    r2 = client.post("/internal/trigger-reviewer-scoring")
    assert r2.status_code == 200
    assert r2.json() == {"recomputed": True}

    # 3) Fetch the scores
    r3 = client.get("/internal/reviewer-scores")
    assert r3.status_code == 200
    scores = r3.json()["scores"]
    assert any(s["reviewer_id"] == "alice" for s in scores)
