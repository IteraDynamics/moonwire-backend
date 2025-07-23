def test_override_suppression(run_local, client, http, base_url):
    payload = {
        "signal_id":      "sig_py_test4",
        "override_reason":"manual override",
        "note":           "pytest override",
        "reviewed_by":    "tester"
    }
    if run_local:
        resp = client.post("/internal/override-suppression", json=payload)
    else:
        resp = http.post(f"{base_url}/internal/override-suppression", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)
    # expect at least a status key
    assert "status" in body or "overridden" in body
