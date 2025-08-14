import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pytest

from src.analytics.source_yield import compute_source_yield


def write_jsonl(path: Path, rows: list):
    path.write_text("\n".join(json.dumps(r) for r in rows))


def make_ts(days_ago: int = 0):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


@pytest.fixture
def tmp_logs(tmp_path):
    flags_path = tmp_path / "retraining_log.jsonl"
    triggers_path = tmp_path / "retraining_triggered.jsonl"
    return flags_path, triggers_path


def test_happy_path(tmp_logs):
    flags_path, triggers_path = tmp_logs
    # twitter: 8 flags, 2 triggers
    # reddit: 4 flags, 1 trigger
    flags = [{"origin": "twitter", "timestamp": make_ts()} for _ in range(8)]
    flags += [{"origin": "reddit", "timestamp": make_ts()} for _ in range(4)]
    triggers = [{"origin": "twitter", "timestamp": make_ts()} for _ in range(2)]
    triggers += [{"origin": "reddit", "timestamp": make_ts()} for _ in range(1)]

    write_jsonl(flags_path, flags)
    write_jsonl(triggers_path, triggers)

    result = compute_source_yield(flags_path, triggers_path, days=7, min_events=1, alpha=0.7)

    assert result["totals"]["flags"] == 12
    assert result["totals"]["triggers"] == 3
    origins = {o["origin"]: o for o in result["origins"]}
    assert origins["twitter"]["trigger_rate"] == 0.25
    assert origins["reddit"]["trigger_rate"] == 0.25
    # Budget should sum to ~100
    total_pct = sum(b["pct"] for b in result["budget_plan"])
    assert abs(total_pct - 100) < 0.01


def test_min_events_filter(tmp_logs):
    flags_path, triggers_path = tmp_logs
    # twitter: 10 flags, reddit: 2 flags
    flags = [{"origin": "twitter", "timestamp": make_ts()} for _ in range(10)]
    flags += [{"origin": "reddit", "timestamp": make_ts()} for _ in range(2)]
    triggers = []

    write_jsonl(flags_path, flags)
    write_jsonl(triggers_path, triggers)

    result = compute_source_yield(flags_path, triggers_path, days=7, min_events=5, alpha=0.5)

    # reddit should not be in budget_plan
    origins_in_plan = {b["origin"] for b in result["budget_plan"]}
    assert "reddit" not in origins_in_plan
    assert "twitter" in origins_in_plan
    # But reddit still in origins list
    origins_all = {o["origin"] for o in result["origins"]}
    assert "reddit" in origins_all


def test_alpha_extremes(tmp_logs):
    flags_path, triggers_path = tmp_logs
    # twitter has high conversion rate, reddit has high volume
    flags = [{"origin": "twitter", "timestamp": make_ts()} for _ in range(5)]
    flags += [{"origin": "reddit", "timestamp": make_ts()} for _ in range(20)]
    triggers = [{"origin": "twitter", "timestamp": make_ts()} for _ in range(3)]
    triggers += [{"origin": "reddit", "timestamp": make_ts()} for _ in range(1)]

    write_jsonl(flags_path, flags)
    write_jsonl(triggers_path, triggers)

    r_alpha1 = compute_source_yield(flags_path, triggers_path, days=7, min_events=1, alpha=1.0)
    r_alpha0 = compute_source_yield(flags_path, triggers_path, days=7, min_events=1, alpha=0.0)

    # alpha=1.0: ranking by trigger rate
    origins_alpha1 = [o["origin"] for o in r_alpha1["budget_plan"]]
    # alpha=0.0: ranking by volume
    origins_alpha0 = [o["origin"] for o in r_alpha0["budget_plan"]]

    assert origins_alpha1[0] == "twitter"
    assert origins_alpha0[0] == "reddit"


def test_missing_triggers_file(tmp_path):
    flags_path = tmp_path / "retraining_log.jsonl"
    triggers_path = tmp_path / "retraining_triggered.jsonl"
    # only flags exist
    flags = [{"origin": "twitter", "timestamp": make_ts()} for _ in range(5)]
    write_jsonl(flags_path, flags)
    # triggers_path intentionally missing

    result = compute_source_yield(flags_path, triggers_path, days=7, min_events=1, alpha=0.5)

    assert result["totals"]["triggers"] == 0
    assert all(o["triggers"] == 0 for o in result["origins"])


def test_windowing_excludes_old(tmp_logs):
    flags_path, triggers_path = tmp_logs
    flags = [
        {"origin": "twitter", "timestamp": make_ts(days_ago=10)},  # too old
        {"origin": "twitter", "timestamp": make_ts(days_ago=1)},   # recent
    ]
    triggers = [{"origin": "twitter", "timestamp": make_ts(days_ago=1)}]

    write_jsonl(flags_path, flags)
    write_jsonl(triggers_path, triggers)

    result = compute_source_yield(flags_path, triggers_path, days=7, min_events=1, alpha=0.5)

    assert result["totals"]["flags"] == 1
    assert result["totals"]["triggers"] == 1


def test_schema_tolerance(tmp_logs):
    flags_path, triggers_path = tmp_logs
    flags = [
        {"source": "twitter", "timestamp": make_ts()},
        {"signal_origin": "reddit", "timestamp": make_ts()},
        {"meta": {"origin": "rss_news"}, "timestamp": make_ts()},
    ]
    triggers = [
        {"source": "twitter", "timestamp": make_ts()},
        {"signal_origin": "reddit", "timestamp": make_ts()},
    ]

    write_jsonl(flags_path, flags)
    write_jsonl(triggers_path, triggers)

    result = compute_source_yield(flags_path, triggers_path, days=7, min_events=1, alpha=0.5)
    origins = {o["origin"] for o in result["origins"]}
    assert {"twitter", "reddit", "rss_news"} <= origins
