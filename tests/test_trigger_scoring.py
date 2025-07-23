def test_trigger_reviewer_scoring(run_local, client, http, base_url):
    if run_local:
        resp = client.post("/internal/trigger-reviewer-scoring")
    else:
        resp = http.post(f"{base_url}/internal/trigger-reviewer-scoring")
    assert resp.status_code == 200
    assert resp.json().get("recomputed") is True
