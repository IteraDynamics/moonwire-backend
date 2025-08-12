# tests/test_signal_origins.py
import json
import time
import pytest
from pathlib import Path
from datetime import datetime, timedelta, timezone

from src.paths import (
    RETRAINING_LOG_PATH,
    RETRAINING_TRIGGERED_LOG_PATH,
)

def _append(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")

def _ts_days_ago(d: int) -> float:
    return (datetime.now(timezone.utc) - timedelta(days=d) + timedelta(seconds=5)).timestamp()

@pytest.fixture
def write_flag_with_origin():
    def _w(signal_id: str, origin: str, ts: float = None):
        _append(
            RETRAINING_LOG_PATH,
            {
                "signal_id": signal_id,
                "origin": origin,
                "timestamp": ts if ts is not None else time.time(),
            },
        )
    return _w

@pytest.fixture
def write_trigger_with_origin():
    def _w(signal_id: str, origin: str, ts: float = None):
        _append(
            RETRAINING_TRIGGERED_LOG_PATH,
            {
                "signal_id": signal_id,
                "origin": origin,
                "timestamp": ts if ts is not None else time.time(),
            },
        )
    return _w

# 1) Happy path (flags only): 3 origins + unknown; pct sum ~100
def test_flags_only_happy_path(client, write_flag_with_origin):
    write_flag_with_origin("s1", "twitter")
    write_flag_with_origin("s2", "reddit")
    write_flag_with_origin("s3", "rss_news")
    write_flag_with_origin("s4", None)  # unknown

    r = client.get("/internal/signal-origins?days=7&include_triggers=false")
    assert r.status_code == 200
    data = r.json()
    assert data["included"]["flags"] == 4
    assert data["included"]["triggers"] == 0
    # ensure all present
    got = {row["origin"]: row["count"] for row in data["origins"]}
    assert got["twitter"] == 1
    assert got["reddit"] == 1
    assert got["rss_news"] == 1
    assert got["unknown"] == 1
    # pct ~ 100
    assert pytest.approx(sum(row["pct"] for row in data["origins"]), rel=1e-3, abs=0.05) == 100.0

# 2) Include triggers: flags+triggers merged; tallies correct
def test_include_triggers(client, write_flag_with_origin, write_trigger_with_origin):
    write_flag_with_origin("s1", "twitter")
    write_trigger_with_origin("s1", "twitter")
    write_flag_with_origin("s2", "reddit")

    r = client.get("/internal/signal-origins?days=7&include_triggers=true")
    assert r.status_code == 200
    data = r.json()
    assert data["included"]["flags"] == 2
    assert data["included"]["triggers"] == 1
    # twitter should be 2 total
    row = next(o for o in data["origins"] if o["origin"] == "twitter")
    assert row["count"] == 2

# 3) min_count filter
def test_min_count_filter(client, write_flag_with_origin):
    write_flag_with_origin("s1", "twitter")
    write_flag_with_origin("s2", "twitter")
    write_flag_with_origin("s3", "reddit")
    r = client.get("/internal/signal-origins?days=7&min_count=2&include_triggers=false")
    assert r.status_code == 200
    data = r.json()
    names = [row["origin"] for row in data["origins"]]
    assert "twitter" in names
    assert "reddit" not in names  # filtered out

# 4) Windowing: exclude older entries
def test_windowing_excludes_old(client, write_flag_with_origin):
    recent = time.time()
    old = _ts_days_ago(30)  # beyond default 7d
    write_flag_with_origin("s1", "twitter", ts=old)
    write_flag_with_origin("s2", "twitter", ts=recent)
    r = client.get("/internal/signal-origins?days=7&include_triggers=false")
    assert r.status_code == 200
    data = r.json()
    row = next(o for o in data["origins"] if o["origin"] == "twitter")
    assert row["count"] == 1  # only recent

# 5) Empty logs → empty list, totals 0
def test_empty_logs(client):
    r = client.get("/internal/signal-origins?days=7")
    assert r.status_code == 200
    data = r.json()
    assert data["total_events"] == 0
    assert data["origins"] == []
    assert data["included"] == {"flags": 0, "triggers": 0}

# 6) Alias mapping: twitter_api & Twitter collapse to twitter
def test_alias_mapping(client, write_flag_with_origin):
    write_flag_with_origin("s1", "twitter_api")
    write_flag_with_origin("s2", "Twitter")
    r = client.get("/internal/signal-origins?days=7&include_triggers=false")
    assert r.status_code == 200
    data = r.json()
    row = next(o for o in data["origins"] if o["origin"] == "twitter")
    assert row["count"] == 2

# 7) Sorting stability: by count desc, then origin asc
def test_sorting_stability(client, write_flag_with_origin):
    for o in ["reddit", "reddit", "rss_news", "twitter", "twitter"]:
        write_flag_with_origin(f"s-{o}-{time.time()}", o)
    r = client.get("/internal/signal-origins?days=7&include_triggers=false")
    assert r.status_code == 200
    names = [row["origin"] for row in r.json()["origins"]]
    # counts: reddit=2, twitter=2, rss_news=1 → tie between reddit/twitter → alphabetic
    assert names[:2] == ["reddit", "twitter"]