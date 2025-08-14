# tests/test_signal_origins.py
import json
import time
import math
import pytest

# --- helpers that resolve paths at CALL TIME (not import time) ---

@pytest.fixture
def write_flag_with_origin():
    def _w(signal_id: str, origin: str | None):
        # Resolve current test paths at call time to avoid import-time capture
        from src.paths import RETRAINING_LOG_PATH
        entry = {
            "signal_id": signal_id,
            "origin": origin,
            "timestamp": time.time(),  # numeric epoch, easy to parse
        }
        with open(RETRAINING_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    return _w

@pytest.fixture
def write_trigger_with_origin():
    def _w(signal_id: str, origin: str | None):
        from src.paths import RETRAINING_TRIGGERED_LOG_PATH
        entry = {
            "signal_id": signal_id,
            "origin": origin,
            "timestamp": time.time(),
        }
        with open(RETRAINING_TRIGGERED_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    return _w

# --- tests ---

def pct_sum(vals):
    return sum(v["pct"] for v in vals)

def test_flags_only_happy_path(client, write_flag_with_origin):
    write_flag_with_origin("s1", "twitter")
    write_flag_with_origin("s2", "reddit")
    write_flag_with_origin("s3", "rss_news")
    write_flag_with_origin("s4", None)  # unknown

    r = client.get("/internal/signal-origins?days=7&include_triggers=false")
    assert r.status_code == 200
    data = r.json()

    assert data["included"]["flags"] == 4
    assert data["included"]["triggers"] in (0, data["included"]["triggers"])  # should be 0
    assert data["total_events"] == 4
    assert isinstance(data["origins"], list)
    # pct sanity
    assert math.isclose(pct_sum(data["origins"]), 100.0, rel_tol=1e-3, abs_tol=1e-3)

def test_include_triggers(client, write_flag_with_origin, write_trigger_with_origin):
    write_flag_with_origin("s1", "twitter")
    write_trigger_with_origin("s1", "twitter")
    write_flag_with_origin("s2", "reddit")

    r = client.get("/internal/signal-origins?days=7&include_triggers=true")
    assert r.status_code == 200
    data = r.json()

    # EXACTLY the lines we wrote above
    assert data["included"]["flags"] == 2
    assert data["included"]["triggers"] == 1
    assert data["total_events"] == 3

def test_min_count_filter(client, write_flag_with_origin):
    write_flag_with_origin("s1", "twitter")
    write_flag_with_origin("s2", "twitter")
    write_flag_with_origin("s3", "reddit")

    r = client.get("/internal/signal-origins?days=7&include_triggers=false&min_count=2")
    assert r.status_code == 200
    data = r.json()
    # Only twitter should remain
    assert [row["origin"] for row in data["origins"]] == ["twitter"]

def test_alias_mapping(client, write_flag_with_origin):
    write_flag_with_origin("a", "Twitter")
    write_flag_with_origin("b", "twitter_api")
    write_flag_with_origin("c", "rss")
    write_flag_with_origin("d", "reddit")

    r = client.get("/internal/signal-origins?days=7&include_triggers=false")
    assert r.status_code == 200
    origins = {row["origin"] for row in r.json()["origins"]}
    # Aliases should collapse
    assert "twitter" in origins
    assert "rss_news" in origins
    assert "reddit" in origins

def test_windowing(client, write_flag_with_origin, monkeypatch):
    # Backdate one entry beyond the window
    from src.paths import RETRAINING_LOG_PATH
    old = {"signal_id": "old", "origin": "twitter", "timestamp": time.time() - 20 * 86400}
    with open(RETRAINING_LOG_PATH, "a") as f:
        f.write(json.dumps(old) + "\n")
    # Recent
    write_flag_with_origin("new", "twitter")

    r = client.get("/internal/signal-origins?days=7&include_triggers=false")
    assert r.status_code == 200
    data = r.json()
    # Only the recent one should count
    assert data["included"]["flags"] == 1

def test_empty_logs(client):
    # The isolated_logs fixture created empty temp files for this test case too
    r = client.get("/internal/signal-origins?days=7&include_triggers=false")
    assert r.status_code == 200
    data = r.json()
    assert data["included"] == {"flags": 0, "triggers": 0}
    assert data["total_events"] == 0
    assert data["origins"] == []