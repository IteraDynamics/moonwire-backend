def test_flag_for_retraining(run_local, client, http, base_url):
    payload = {
        "signal_id": "sig_py_test3",
        "reason":    "model_update_needed",
        "note":      "pytest retrain",
    }
    if run_local:
        resp = client.post("/internal/flag-for-retraining", json=payload)
    else:
        resp = http.post(f"{base_url}/internal/flag-for-retraining", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    # should at least return a status or similar
    assert isinstance(body, dict)
    assert "status" in body or "retrained" in body
