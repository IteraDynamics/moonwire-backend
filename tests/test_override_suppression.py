def test_override_suppression(client):
    payload = {
        "signal_id": "sig_override",
        "override_reason": "unit test",
    }
    r = client.post("/internal/override-suppression", json=payload)
    assert r.status_code == 200
    # stub returns {}, 200
