# tests/test_reviewer_scores.py

def test_get_reviewer_scores(client):
    """
    Simply verifies that GET /internal/reviewer-scores returns a 200
    and that the JSON has a 'scores' key containing a list.
    """
    response = client.get("/internal/reviewer-scores")
    assert response.status_code == 200

    body = response.json()
    assert "scores" in body, "Response JSON must include a 'scores' field"
    assert isinstance(body["scores"], list), "'scores' should be a list"
