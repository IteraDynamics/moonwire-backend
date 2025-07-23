def test_log_signal_for_review(run_local, client, http, base_url):
    payload = {
        "signal_id": "sig_py_test1",
        "asset":     "TASK2XYZ",
        "trust_score": 0.20,
        "suppression_reason": "trust_score_below_threshold",
    }
    if run_local:
        resp = client.post("/internal/log-signal-for-review", json=payload)
    else:
        resp = http.post(f"{base_url}/internal/log-signal-for-review", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)
    # Expect at least a status key
    assert "status" in body
