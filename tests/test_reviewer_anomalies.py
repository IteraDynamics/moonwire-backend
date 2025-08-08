# tests/test_reviewer_anomalies.py

import json
import pytest

def test_filters_on_min_score(client, write_score):
    # r_lo scores below threshold, r_hi above
    write_score("r_lo1", 0.40)
    write_score("r_lo2", 0.55)
    write_score("r_hi", 0.80)

    r = client.get("/internal/reviewer-anomalies")  # default min_score=0.6
    assert r.status_code == 200
    data = r.json()["anomalies"]
    ids = [row["reviewer_id"] for row in data]
    assert "r_lo1" in ids and "r_lo2" in ids
    assert "r_hi" not in ids
    # Sorted ascending by score: r_lo1 (0.40) should come before r_lo2 (0.55)
    assert ids[:2] == ["r_lo1", "r_lo2"]

def test_filters_on_score_change_drop(client, write_score):
    # Start high then drop by >= 0.15 → should be included even if still >= min_score
    write_score("r_drop", 0.90)
    write_score("r_drop", 0.70)  # -0.20

    r = client.get("/internal/reviewer-anomalies")  # min_score=0.6, current score 0.70
    assert r.status_code == 200
    data = r.json()["anomalies"]
    row = next(x for x in data if x["reviewer_id"] == "r_drop")
    assert pytest.approx(row["score_change"], rel=1e-6) == -0.20

def test_sorted_and_limit(client, write_score):
    write_score("a", 0.20)
    write_score("b", 0.30)
    write_score("c", 0.40)
    write_score("d", 0.10)

    r = client.get("/internal/reviewer-anomalies?limit=3")
    assert r.status_code == 200
    ids = [row["reviewer_id"] for row in r.json()["anomalies"]]
    # Sorted worst-first (lowest score first), limited to 3
    assert ids == ["d", "a", "b"]

def test_graceful_missing_fields(client):
    # Inject a malformed entry with missing score; should not blow up
    import importlib, src.paths
    importlib.reload(src.paths)
    from src.paths import REVIEWER_SCORES_PATH

    with REVIEWER_SCORES_PATH.open("a") as f:
        f.write(json.dumps({"reviewer_id": "bad_entry_no_score"}) + "\n")

    r = client.get("/internal/reviewer-anomalies")
    assert r.status_code == 200
    # Just ensure it returns a valid payload (may be empty depending on other tests)
    assert "anomalies" in r.json()

def test_empty_file_returns_empty(client):
    # Fresh test env has empty reviewer_scores.jsonl via fixture
    r = client.get("/internal/reviewer-anomalies")
    assert r.status_code == 200
    assert r.json() == {"anomalies": []}