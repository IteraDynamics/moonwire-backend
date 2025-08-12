# tests/test_demo_seed.py

import os
from datetime import datetime, timedelta, timezone

from scripts.mw_demo_summary import generate_demo_data_if_needed
from src.demo_seed import seed_reviewers_if_empty


def test_seed_reviewers_if_empty_generates_in_range():
    now = datetime(2025, 8, 8, 12, 0, 0, tzinfo=timezone.utc)
    reviewers, events = seed_reviewers_if_empty([], now=now)
    assert 3 <= len(reviewers) <= 5
    assert len(events) == len(reviewers)
    for r in reviewers:
        assert r["weight"] in (0.75, 1.0, 1.25)
        assert r["id"].startswith("demo-")
    # timestamps within last 60 min
    for e in events:
        ts = datetime.fromisoformat(e["timestamp"])
        assert timedelta(minutes=0) < (now - ts) <= timedelta(minutes=60)


def test_generate_demo_data_demo_off_returns_empty(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    reviewers, events = generate_demo_data_if_needed([])
    assert reviewers == []
    assert events == []


def test_generate_demo_data_demo_on_and_empty_seeds(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    reviewers, events = generate_demo_data_if_needed([])
    assert 3 <= len(reviewers) <= 5
    assert len(events) == len(reviewers)


def test_generate_demo_data_demo_on_but_real_reviewers_no_seed(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    real = [{"id": "alice", "weight": 1.0}]
    reviewers, events = generate_demo_data_if_needed(real)
    assert reviewers == real
    assert events == []