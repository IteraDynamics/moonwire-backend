# tests/test_trigger_scoring.py

def test_trigger_reviewer_scoring(client):
    """
    Verify that POST /internal/trigger-reviewer-scoring returns 200
    and a JSON body containing {"recomputed": True}.
    """
    response = client.post("/internal/trigger-reviewer-scoring")
    assert response.status_code == 200

    data = response.json()
    assert "recomputed" in data, "Response JSON must include 'recomputed'"
    assert data["recomputed"] is True
