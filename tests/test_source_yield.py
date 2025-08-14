#!/usr/bin/env python3
"""
Tests for /internal/source-yield-plan endpoint and compute_source_yield().
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from src.analytics.source_yield import compute_source_yield


@pytest.fixture
def tmp_logs(tmp_path):
    flags_path = tmp_path / "retraining_log.jsonl"
    triggers_path = tmp_path / "retraining_triggered.jsonl"

    now = datetime.now(timezone.utc)

    def write_jsonl(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # Flags: twitter(6), reddit(3), rss_news(1)
    flags = []
    for i in range(6):
        flags.append({"origin": "twitter", "timestamp": (now - timedelta(days=1)).isoformat()})
    for i in range(3):
        flags.append({"origin": "reddit", "timestamp": (now - timedelta(days=2)).isoformat()})
    for i in range(1):
        flags.append({"origin": "rss_news", "timestamp": (now - timedelta(days=2)).isoformat()})
    write_jsonl(flags_path, flags)

    # Triggers: twitter(2), reddit(1)
    triggers = []
    for i in range(2):
        triggers.append({"origin": "twitter", "timestamp": (now - timedelta(hours=10)).isoformat()})
    for i in range(1):
        triggers.append({"origin": "reddit", "timestamp": (now - timedelta(hours=5)).isoformat()})
    write_jsonl(triggers_path, triggers)

    return flags_path, triggers_path


def test_happy_path(tmp_logs):
    flags_path, triggers_path = tmp_logs
    res = compute_source_yield(flags_path, triggers_path, days=7, min_events=2, alpha=0.7)

    origins = {o["origin"]: o for o in res["origins"]}
    assert origins["twitter"]["flags"] == 6
    assert origins["twitter"]["triggers"] == 2
    assert abs(origins["twitter"]["trigger_rate"] - (2/6)) < 1e-6

    # Budget plan should sum to ~100
    total_pct = sum(p["pct"] for p in res["budget_plan"])
    assert 99.9 <= total_pct <= 100.1

    # Order: twitter first
    assert res["budget_plan"][0]["origin"] == "twitter"


def test_min_events_filter(tmp_logs):
    flags_path, triggers_path = tmp_logs
    res = compute_source_yield(flags_path, triggers_path, days=7, min_events=5, alpha=0.7)
    # rss_news only has 1 flag, should be excluded from budget_plan
    origins_in_plan = {p["origin"] for p in res["budget_plan"]}
    assert "rss_news" not in origins_in_plan
    # But rss_news still present in raw origins
    origins_all = {o["origin"] for o in res["origins"]}
    assert "rss_news" in origins_all


def test_alpha_extremes(tmp_logs):
    flags_path, triggers_path = tmp_logs
    res_vol = compute_source_yield(flags_path, triggers_path, days=7, min_events=1, alpha=0.0)
    res_conv = compute_source_yield(flags_path, triggers_path, days=7, min_events=1, alpha=1.0)
    # Rankings should differ between pure volume and pure conversion
    plan_vol = [p["origin"] for p in res_vol["budget_plan"]]
    plan_conv = [p["origin"] for p in res_conv["budget_plan"]]
    assert plan_vol != plan_conv


def test_missing_triggers_file(tmp_path):
    flags_path = tmp_path / "retraining_log.jsonl"
    triggers_path = tmp_path / "missing.jsonl"

    now = datetime.now(timezone.utc)
    flags = [
        {"origin": "twitter", "timestamp": now.isoformat()},
        {"origin": "reddit", "timestamp": now.isoformat()},
    ]
    with open(flags_path, "w") as f:
        for r in flags:
            f.write(json.dumps(r) + "\n")

    res = compute_source_yield(flags_path, triggers_path, days=7, min_events=1, alpha=0.7)
    for o in res["origins"]:
        assert o["trigger_rate"] == 0.0  # no triggers file


def test_windowing_excludes_old_events(tmp_path):
    flags_path = tmp_path / "retraining_log.jsonl"
    triggers_path = tmp_path / "retraining_triggered.jsonl"

    now = datetime.now(timezone.utc)
    old_time = now - timedelta(days=30)

    flags = [
        {"origin": "twitter", "timestamp": now.isoformat()},
        {"origin": "twitter", "timestamp": old_time.isoformat()},
    ]
    with open(flags_path, "w") as f:
        for r in flags:
            f.write(json.dumps(r) + "\n")

    with open(triggers_path, "w") as f:
        pass  # empty

    res = compute_source_yield(flags_path, triggers_path, days=7, min_events=1, alpha=0.7)
    origins = {o["origin"]: o for o in res["origins"]}
    assert origins["twitter"]["flags"] == 1  # only recent one counted
