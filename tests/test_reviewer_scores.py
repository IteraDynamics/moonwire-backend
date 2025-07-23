def test_get_reviewer_scores(run_local, client, http, base_url):
    if run_local:
        resp = client.get("/internal/reviewer-scores")
    else:
        resp = http.get(f"{base_url}/internal/reviewer-scores")
    assert resp.status_code == 200
    body = resp.json()
    assert "scores" in body
    assert isinstance(body["scores"], list)
