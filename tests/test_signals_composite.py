def test_signals_composite(client):
    # minimal valid query
    r = client.get("/signals/composite?asset=BTC&twitter_score=0.1&news_score=0.2")
    assert r.status_code == 200
    data = r.json()
    # should be a dict with at least one key
    assert isinstance(data, dict)
    assert len(data) > 0
