def test_debug_jsonl_status(run_local, client, http, base_url):
    if run_local:
        resp = client.get("/internal/debug/jsonl-status")
    else:
        resp = http.get(f"{base_url}/internal/debug/jsonl-status")
    assert resp.status_code == 200
    status = resp.json()
    # both files must be present in the status
    for key in ("reviewer_impact_log", "reviewer_scores"):
        assert key in status
        info = status[key]
        assert isinstance(info["exists"], bool)
        assert isinstance(info["size_bytes"], int)
        assert isinstance(info["writable"], bool)
        assert isinstance(info["absolute_path"], str)
