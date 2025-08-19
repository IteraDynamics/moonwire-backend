import json, os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.analytics.burst_detection import compute_bursts
from main import app
import src.paths as paths

def _j(d): return json.dumps(d, separators=(",", ":"))
def _log(ts, origin): return {"timestamp": ts.isoformat(), "origin": origin}

def test_burst_detection_happy(tmp_path: Path):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=23)  # 24 buckets window

    # Baseline: 1 event/hour for first 10 hours; spike of 10 at hour 12
    flags = []
    for i in range(10):
        ts = start + timedelta(hours=i)
        flags.append(_log(ts, "twitter"))
    spike_ts = start + timedelta(hours=12)
    for _ in range(10):
        flags.append(_log(spike_ts, "twitter"))

    fpath = tmp_path / "flags.jsonl"; fpath.write_text("\n".join(_j(r) for r in flags))
    tpath = tmp_path / "triggers.jsonl"; tpath.write_text("")

    out = compute_bursts(fpath, tpath, days=1, interval="hour", z_thresh=2.0)
    tw = next((o for o in out["origins"] if o["origin"] == "twitter"), {"bursts": []})
    times = [b["timestamp_bucket"] for b in tw["bursts"]]
    assert spike_ts.isoformat().replace("+00:00","Z") in times

def test_burst_detection_no_bursts(tmp_path: Path):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=23)

    flags = []
    # flat-ish series: 1 event at hour 0..5
    for i in range(6):
        flags.append(_log(start + timedelta(hours=i), "reddit"))

    fpath = tmp_path / "flags.jsonl"; fpath.write_text("\n".join(_j(r) for r in flags))
    tpath = tmp_path / "triggers.jsonl"; tpath.write_text("")

    out = compute_bursts(fpath, tpath, days=1, interval="hour", z_thresh=3.0)
    rd = next((o for o in out["origins"] if o["origin"] == "reddit"), {"bursts": []})
    assert rd["bursts"] == []

def test_burst_detection_api_demo_seed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path)
    (tmp_path / "retraining_log.jsonl").write_text("")
    (tmp_path / "retraining_triggered.jsonl").write_text("")

    old = os.environ.get("DEMO_MODE")
    os.environ["DEMO_MODE"] = "true"
    try:
        client = TestClient(app)
        res = client.get("/internal/burst-detection?days=7&interval=hour&z_thresh=2.0")
        assert res.status_code == 200
        data = res.json()
        # Should have seeded bursts
        origins = data.get("origins", [])
        assert origins
        assert any(o.get("bursts") for o in origins)
    finally:
        if old is None:
            del os.environ["DEMO_MODE"]
        else:
            os.environ["DEMO_MODE"] = old

def test_burst_detection_param_validation(tmp_path: Path):
    client = TestClient(app)
    # Bad interval
    res = client.get("/internal/burst-detection?interval=minute")
    assert res.status_code == 422
