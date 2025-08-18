import json, os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.analytics.lead_lag import compute_lead_lag
from main import app
import src.paths as paths

def _j(d): return json.dumps(d, separators=(",", ":"))
def _log(ts, origin): return {"timestamp": ts.isoformat(), "origin": origin}

def _write_series(buf: list, start: datetime, origin: str, counts: list[int], step_hours=1):
    for i, c in enumerate(counts):
        ts = start + timedelta(hours=i * step_hours)
        for _ in range(int(c)):
            buf.append(_log(ts, origin))

def test_lead_lag_happy_positive(tmp_path: Path):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # base shape over 10 consecutive hours
    base = [0,1,3,5,4,3,2,1,0,1]
    twitter = base[:]                  # A
    reddit  = [0,0,0] + base[:-3]      # B delayed 3h (A leads B by +3)

    flags = []
    _write_series(flags, now - timedelta(hours=9), "twitter", twitter)
    _write_series(flags, now - timedelta(hours=9), "reddit",  reddit)

    fpath = tmp_path / "flags.jsonl"; fpath.write_text("\n".join(_j(r) for r in flags))
    tpath = tmp_path / "triggers.jsonl"; tpath.write_text("")

    out = compute_lead_lag(fpath, tpath, days=2, interval="hour", max_lag=6, use="flags")
    pairs = { (p["a"], p["b"]): p for p in out["pairs"] }

    pr = pairs.get(("twitter", "reddit"))
    assert pr is not None
    assert pr["best_lag"] == 3
    assert pr["leader"] == "twitter"
    assert pr["correlation"] > 0.6

def test_lead_lag_negative_lag(tmp_path: Path):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    base = [0,2,4,6,4,2,1,0]
    reddit    = base[:]                 # B
    rss_news  = [0,0] + base[:-2]       # A delayed 2h relative to B (B leads A)
    # For pair (A=rss_news, B=reddit), best_lag should be -2 and leader=reddit.

    flags = []
    _write_series(flags, now - timedelta(hours=7), "rss_news", rss_news)  # A
    _write_series(flags, now - timedelta(hours=7), "reddit",   reddit)    # B

    fpath = tmp_path / "flags.jsonl"; fpath.write_text("\n".join(_j(r) for r in flags))
    tpath = tmp_path / "triggers.jsonl"; tpath.write_text("")

    out = compute_lead_lag(fpath, tpath, days=2, interval="hour", max_lag=6, use="flags")
    pairs = { (p["a"], p["b"]): p for p in out["pairs"] }

    pr = pairs.get(("rss_news", "reddit"))  # A=rss_news, B=reddit
    assert pr is not None
    assert pr["best_lag"] == -2
    assert pr["leader"] == "reddit"
    assert abs(pr["correlation"]) > 0.6

def test_lead_lag_skips_constant_and_low_overlap(tmp_path: Path):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # Constant series (flat)
    flat = [3,3,3,3,3]
    var  = [0,1,2,3,4]

    flags = []
    _write_series(flags, now - timedelta(hours=4), "flat", flat)
    _write_series(flags, now - timedelta(hours=4), "var",  var)

    # Another pair with only 2 overlapping aligned points at lag=3 -> should be skipped (<3)
    a = [1,0,0,0,0]
    b = [0,0,0,0,1]
    _write_series(flags, now - timedelta(hours=4), "a", a)
    _write_series(flags, now - timedelta(hours=4), "b", b)

    fpath = tmp_path / "flags.jsonl"; fpath.write_text("\n".join(_j(r) for r in flags))
    tpath = tmp_path / "triggers.jsonl"; tpath.write_text("")

    out = compute_lead_lag(fpath, tpath, days=2, interval="hour", max_lag=6, use="flags")

    # flat should not appear in any pair
    for p in out["pairs"]:
        assert p["a"] != "flat" and p["b"] != "flat"

    # (a,b) pair should not appear due to insufficient overlap
    assert ("a", "b") not in {(p["a"], p["b"]) for p in out["pairs"]}

def test_lead_lag_api_demo_seed(tmp_path: Path, monkeypatch):
    # empty logs + DEMO_MODE => seeded results
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path)
    (tmp_path / "retraining_log.jsonl").write_text("")
    (tmp_path / "retraining_triggered.jsonl").write_text("")

    old = os.environ.get("DEMO_MODE")
    os.environ["DEMO_MODE"] = "true"
    try:
        client = TestClient(app)
        res = client.get("/internal/lead-lag?days=7&interval=hour&max_lag=24&top=3&use=flags")
        assert res.status_code == 200
        data = res.json()
        assert data["pairs"]
        for p in data["pairs"]:
            assert 0.3 <= abs(p["correlation"]) <= 0.8
            assert abs(int(p["best_lag"])) <= 24
    finally:
        if old is None:
            del os.environ["DEMO_MODE"]
        else:
            os.environ["DEMO_MODE"] = old

def test_lead_lag_param_validation(tmp_path: Path, monkeypatch):
    client = TestClient(app)
    # Bad interval -> 422
    res = client.get("/internal/lead-lag?interval=minute")
    assert res.status_code == 422
    # Bad use -> 422
    res = client.get("/internal/lead-lag?use=both")
    assert res.status_code == 422
