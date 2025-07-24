def test_log_signal_for_review(client):
    payload = {
        "signal_id": "sig_review",
        "asset": "BTC",
        "trust_score": 0.5,
        "suppression_reason": "unit test",
    }
    r = client.post("/internal/log-signal-for-review", json=payload)
    assert r.status_code == 200
    # stub returns {}, 200
