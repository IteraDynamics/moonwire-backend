def test_flag_for_retraining(client):
    payload = {
        "signal_id": "sig_for_retrain",
        "reason": "unit test",
    }
    r = client.post("/internal/flag-for-retraining", json=payload)
    assert r.status_code == 200
    # No error means our stub ran
