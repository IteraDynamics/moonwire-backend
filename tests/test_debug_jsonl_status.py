def test_debug_jsonl_status_local(client):
    r = client.get("/internal/debug/jsonl-status")
    assert r.status_code == 200
    data = r.json()
    assert "reviewer_impact_log" in data
    assert "reviewer_scores" in data
    assert data["reviewer_impact_log"]["exists"] is True
