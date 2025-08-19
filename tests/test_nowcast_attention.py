import json, os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.analytics.nowcast_attention import compute_nowcast_attention
from main import app
import src.paths as paths

def _j(d): return json.dumps(d, separators=(",", ":"))
def _log(ts, origin): return {"timestamp": ts.isoformat(), "origin": origin}

def _write_series(buf: list, start: datetime, origin: str, counts: list[int], step="hour"):
    step_td = timedelta(hours=1) if step == "hour" else timedelta(days=1)
    for i, c in enumerate(counts):
        ts = start + i * step_td
        for _ in range(int(c)):
            buf.append(_log(ts, origin))

def test_nowcast_ranking_happy(tmp_path: Path):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=23)

    flags = []
    # twitter: spike at the last bucket -> high z
    tw = [0]*22 + [1, 10]
    # reddit: flat-ish -> low z
    rd = [1]*24
    # rss: moderate
    rs = [0]*20 + [1,2,1,1]

    _write_series(flags, start, "twitter", tw)
    _write_series(flags, start, "reddit", rd)
    _write_series(flags, start, "rss_news", rs)

    fpath = tmp_path / "flags.jsonl"; fpath.write_text("\n".join(_j(r) for r in flags))
    tpath = tmp_path / "triggers.jsonl"; tpath.write_text("")

    out = compute_nowcast_attention(fpath, tpath, days=1, interval="hour", lookback=12, z_cap=5.0, top=3)
    rows = out["origins"]
    assert rows, "should produce attention scores"
    # twitter should rank #1 due to spike
    assert rows[0]["origin"] == "twitter"

def test_nowcast_api_demo_seed(tmp_path: Path, monkeypatch):
    # point to empty logs -> demo seed
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path)
    (tmp_path / "retraining_log.jsonl").write_text("")
    (tmp_path / "retraining_triggered.jsonl").write_text("")

    old = os.environ.get("DEMO_MODE"); os.environ["DEMO_MODE"] = "true"
    try:
        client = TestClient(app)
        res = client.get("/internal/nowcast-attention?days=7&interval=hour&top=3")
        assert res.status_code == 200
        data = res.json()
        assert data["origins"] and len(data["origins"]) <= 3
        # scores should be present
        assert "score" in data["origins"][0]
    finally:
        if old is None: del os.environ["DEMO_MODE"]
        else: os.environ["DEMO_MODE"] = old

def test_nowcast_param_validation():
    client = TestClient(app)
    res = client.get("/internal/nowcast-attention?interval=minute")
    assert res.status_code == 422
