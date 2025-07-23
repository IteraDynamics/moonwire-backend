def test_reviewer_impact_log_and_status(run_local, client, http, base_url):
    payload = {
        "signal_id":   "sig_py_test2",
        "reviewer_id": "rev_pytest",
        "action":      "override",
        "trust_delta": 0.1,
        "note":        "pytest automation",
    }
    # 1) Write to the impact log
    if run_local:
        resp = client.post("/internal/reviewer-impact-log", json=payload)
    else:
        resp = http.post(f"{base_url}/internal/reviewer-impact-log", json=payload)
    assert resp.status_code == 200
    assert resp.json().get("status") == "logged"

    # 2) Confirm file status
    if run_local:
        resp2 = client.get("/internal/debug/jsonl-status")
    else:
        resp2 = http.get(f"{base_url}/internal/debug/jsonl-status")
    assert resp2.status_code == 200
    status = resp2.json()["reviewer_impact_log"]
    assert status["exists"] is True
    assert status["size_bytes"] > 0
    assert status["writable"] is True
