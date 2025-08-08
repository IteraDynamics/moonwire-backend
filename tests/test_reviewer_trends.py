# tests/test_reviewer_trends.py

import time
import pytest
from datetime import timedelta, datetime

def test_returns_subset_for_days(client, write_score_history):
    now = time.time()
    # within 30 days
    write_score_history("alice", 0.80, ts=now - 5 * 86400)
    write_score_history("alice", 0.85, ts=now - 1 * 86400)
    # outside window
    write_score_history("alice", 0.70, ts=now - 60 * 86400)

    r = client.get("/internal/reviewer-trends/alice?days=30")
    assert r.status_code == 200
    data = r.json()
    scores = [pt["score"] for pt in data["trend_data"]]
    assert scores == [0.80, 0.85]  # only within 30d, sorted asc

def test_change_over_period(client, write_score_history):
    now = time.time()
    write_score_history("bob", 0.60, ts=now - 4 * 86400)
    write_score_history("bob", 0.75, ts=now - 2 * 86400)
    write_score_history("bob", 0.72, ts=now - 1 * 86400)

    r = client.get("/internal/reviewer-trends/bob?days=30")
    assert r.status_code == 200
    data = r.json()
    assert data["change_over_period"] == pytest.approx(0.72 - 0.60)

def test_missing_history_file(client, write_score, tmp_path, monkeypatch):
    # Remove history file to simulate missing
    import importlib, src.paths
    importlib.reload(src.paths)
    from src.paths import REVIEWER_SCORES_HISTORY_PATH, REVIEWER_SCORES_PATH
    if REVIEWER_SCORES_HISTORY_PATH.exists():
        REVIEWER_SCORES_HISTORY_PATH.unlink()

    # Still should return current_score if snapshot exists
    write_score("charlie", 0.88)

    r = client.get("/internal/reviewer-trends/charlie")
    assert r.status_code == 200
    data = r.json()
    assert data["trend_data"] == []
    assert data["current_score"] == pytest.approx(0.88)

def test_missing_reviewer_graceful(client):
    r = client.get("/internal/reviewer-trends/ghost")
    assert r.status_code == 200
    data = r.json()
    assert data["reviewer_id"] == "ghost"
    assert data["trend_data"] == []
    assert data["current_score"] is None
    assert data["change_over_period"] == 0.0