# tests/test_reviewer_leaderboard.py

import pytest
from pathlib import Path

def test_sorted_high_to_low(client, write_score):
    write_score("r1", 0.92)  # -> 1.25
    write_score("r2", 0.60)  # -> 1.0
    write_score("r3", 0.40)  # -> 0.75

    r = client.get("/internal/reviewer-leaderboard")
    assert r.status_code == 200
    data = r.json()["leaderboard"]
    ids = [row["reviewer_id"] for row in data]
    assert ids == ["r1", "r2", "r3"]  # high→low by banded weight then score
    # sanity on weights
    weights = [row["weight"] for row in data]
    assert weights == [1.25, 1.0, 0.75]

def test_limit_param_respected(client, write_score):
    for i in range(5):
        write_score(f"r{i}", 0.9 - i * 0.1)
    r = client.get("/internal/reviewer-leaderboard", params={"limit": 2})
    assert r.status_code == 200
    assert len(r.json()["leaderboard"]) == 2

def test_empty_file_returns_empty(client):
    # isolated_logs fixture creates empty reviewer_scores.jsonl
    r = client.get("/internal/reviewer-leaderboard")
    assert r.status_code == 200
    assert r.json() == {"leaderboard": []}

def test_mixed_missing_fields(client, write_score, tmp_path, monkeypatch):
    # Write some valid scores
    write_score("ok1", 0.8)     # -> 1.25
    write_score("ok2", 0.55)    # -> 1.0

    # Manually append an entry missing score (falls back to weight=1.0)
    from src.paths import REVIEWER_SCORES_PATH
    path = Path(REVIEWER_SCORES_PATH)
    with path.open("a") as f:
        f.write('{"reviewer_id":"no_score","timestamp": 1723011111}\n')

    r = client.get("/internal/reviewer-leaderboard", params={"limit": 10})
    assert r.status_code == 200
    rows = r.json()["leaderboard"]

    # Ensure all present and sorted by weight desc
    ids = [row["reviewer_id"] for row in rows]
    assert "ok1" in ids and "ok2" in ids and "no_score" in ids

    weights = {row["reviewer_id"]: row["weight"] for row in rows}
    assert weights["ok1"] == 1.25
    assert weights["ok2"] == 1.0
    assert weights["no_score"] == 1.0  # fallback

    # last_updated is allowed to be None or ISO; just assert key exists
    assert "last_updated" in rows[0]
