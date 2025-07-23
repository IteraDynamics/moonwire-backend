def test_signals_composite(run_local, client, http, base_url):
    params = {
        "asset": "TASK2XYZ",
        "twitter_score": 0.20,
        "news_score":  0.15,
    }
    if run_local:
        resp = client.get("/signals/composite", params=params)
    else:
        resp = http.get(f"{base_url}/signals/composite", params=params)

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    # optionally assert expected fields:
    # assert "signal_id" in data
    # assert "asset"     in data
    # assert "composite_score" in data
