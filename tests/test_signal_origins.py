# tests/test_signal_origins.py
import json
import time
from pathlib import Path

import pytest

from src.paths import RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH


# --- helpers -----------------------------------------------------

def _append_jsonl(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(obj) + "\n")


@pytest.fixture
def write_flag_with_origin():
    def _w(signal_id: str, origin: str | None, ts: float | None = None):
        _append_jsonl(Path(RETRAINING_LOG_PATH), {
            "signal_id": signal_id,
            "origin": origin,
            "timestamp": ts if ts is not None else time.time(),
        })
    return _w


@pytest.fixture
def write_trigger_with_origin():
    def _w(signal_id: str, origin: str | None, ts: float | None = None):
        _append_jsonl(Path(RETRAINING_TRIGGERED_LOG_PATH), {
            "signal_id": signal_id,
            "origin": origin,
            "timestamp": ts if ts is not None else time.time(),
        })
    return _w


# --- tests -------------------------------------------------------

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
    assert data["total_events"] == 4

    # expect origins present and sorted by count desc then origin asc
    origins = data["origins"]
    names = [o["origin"] for o in origins]
    counts = [o["count"] for o in origins]
    assert set(names) == {"twitter", "reddit", "rss_news", "unknown"}
    assert sum(counts) == 4
    # percentages sum ~ 100
    if data["total_events"] > 0:
        pct_sum = sum(o["pct"] for o in origins)
        assert abs(pct_sum - 100.0) <= 0.1


def test_include_triggers(client, write_flag_with_origin, write_trigger_with_origin):
    write_flag_with_origin("s1", "twitter")
    write_trigger_with_origin("s1", "twitter")
    write_flag_with_origin("s2", "reddit")

    r = client.get("/internal/signal-origins?days=7&include_triggers=true")
    assert r.status_code == 200
    data = r.json()

    assert data["included"]["flags"] == 2
    assert data["included"]["triggers"] == 1
    assert data["total_events"] == 3

    # twitter should have 2, reddit 1 (counts merged from flags+triggers)
    by_origin = {o["origin"]: o for o in data["origins"]}
    assert by_origin["twitter"]["count"] == 2
    assert by_origin["reddit"]["count"] == 1


def test_min_count_filters_after_aggregation(client, write_flag_with_origin):
    # twitter x3, reddit x1, unknown x1
    for _ in range(3):
        write_flag_with_origin("t", "twitter")
    write_flag_with_origin("r", "reddit")
    write_flag_with_origin("u", None)

    r = client.get("/internal/signal-origins?days=7&include_triggers=false&min_count=2")
    assert r.status_code == 200
    data = r.json()

    names = [o["origin"] for o in data["origins"]]
    assert "twitter" in names
    assert "reddit" not in names
    assert "unknown" not in names

    # totals unchanged
    assert data["included"]["flags"] == 5
    assert data["total_events"] == 5


def test_windowing_excludes_old(client, write_flag_with_origin):
    now = time.time()
    old = now - 20 * 24 * 3600  # 20 days ago
    write_flag_with_origin("s_old", "twitter", ts=old)
    write_flag_with_origin("s_new", "reddit", ts=now)

    r = client.get("/internal/signal-origins?days=7&include_triggers=false")
    assert r.status_code == 200
    data = r.json()

    # Only the recent one should count
    assert data["included"]["flags"] == 1
    names = [o["origin"] for o in data["origins"]]
    assert names == ["reddit"]  # single item list


def test_alias_mapping_and_sorting(client, write_flag_with_origin):
    # twitter_api and Twitter collapse to "twitter"; also unknown
    write_flag_with_origin("a", "twitter_api")
    write_flag_with_origin("b", "Twitter")
    write_flag_with_origin("c", "rss")
    write_flag_with_origin("d", None)

    r = client.get("/internal/signal-origins?days=7&include_triggers=false")
    assert r.status_code == 200
    data = r.json()

    by_origin = {o["origin"]: o for o in data["origins"]}
    assert by_origin["twitter"]["count"] == 2
    assert by_origin["rss_news"]["count"] == 1
    assert by_origin["unknown"]["count"] == 1


def test_empty_logs_returns_zeros(client):
    r = client.get("/internal/signal-origins?days=7&include_triggers=true")
    assert r.status_code == 200
    data = r.json()
    assert data["origins"] == []
    assert data["included"] == {"flags": 0, "triggers": 0}
    assert data["total_events"] == 0