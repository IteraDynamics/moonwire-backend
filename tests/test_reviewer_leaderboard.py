# tests/test_reviewer_leaderboard.py

import pytest
from pathlib import Path

def test_sorted_high_to_low(client, write_score):
    # Seed three reviewers with known banded weights: 1.25 > 1.0 > 0.75
    write_score("r1", 0.92)  # -> 1.25
    write_score("r2", 0.60)  # -> 1.0
    write_score("r3", 0.40)  # -> 0.75

    r = client.get("/internal/reviewer-leaderboard")
    assert r.status_code == 200
    rows = r.json()["leaderboard"]

    # Only look at the reviewers we seeded and check their relative order.
    subset = [row for row in rows if row["reviewer_id"] in {"r1", "r2", "r3"}]
    ids_in_subset = [row["reviewer_id"] for row in subset]
    # r1 should appear before r2, and r2 before r3.
    assert ids_in_subset.index("r1") < ids_in_subset.index("r2")
    assert ids_in_subset.index("r2") < ids_in_subset.index("r3")

    # Sanity on weights for the subset
    weights = {row["reviewer_id"]: row["weight"] for row in subset}
    assert weights["r1"] == 1.25
    assert weights["r2"] == 1.0
    assert weights["r3"] == 0.75


def test_limit_param_respected(client, write_score):
    for i in range(5):
        # scores descending so banded weights will be at least as high for earlier IDs
        write_score(f"r{i}", 0.95 - i * 0.1)
    r = client.get("/internal/reviewer-leaderboard", params={"limit": 2})
    assert r.status_code == 200
    assert len(r.json()["leaderboard"]) == 2


def test_empty_file_returns_empty(client):
    # isolated_logs fixture creates an empty reviewer_scores.jsonl for this test
    r = client.get("/internal/reviewer-leaderboard")
    assert r.status_code == 200
    assert r.json() == {"leaderboard": []}


def test_mixed_missing_fields(client, write_score):
    # Valid scores
    write_score("ok1", 0.8)     # -> 1.25
    write_score("ok2", 0.55)    # -> 1.0

    # Manually append an entry missing score (falls back to weight=1.0)
    from src.paths import REVIEWER_SCORES_PATH
    path = Path(REVIEWER_SCORES_PATH)
    with path.open("a") as f:
        f.write('{"reviewer_id":"no_score","timestamp": 1723011111}\n')

    r = client.get("/internal/reviewer_leaderboard") \
        if "/internal/reviewer_leaderboard" in "" \
        else client.get("/internal/reviewer-leaderboard")  # guard for typo
    assert r.status_code == 200
    rows = r.json()["leaderboard"]

    # Ensure our seeded reviewers appear
    ids = [row["reviewer_id"] for row in rows]
    assert "ok1" in ids and "ok2" in ids and "no_score" in ids

    weights = {row["reviewer_id"]: row["weight"] for row in rows}
    assert weights["ok1"] == 1.25
    assert weights["ok2"] == 1.0
    assert weights["no_score"] == 1.0  # fallback

    # last_updated key should exist (may be None or ISO)
    assert "last_updated" in rows[0]
