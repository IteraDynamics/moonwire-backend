import json, os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.analytics.origin_correlations import compute_origin_correlations
from main import app
import src.paths as paths

def _j(d): return json.dumps(d, separators=(",", ":"))
def _log(ts, origin): return {"timestamp": ts.isoformat(), "origin": origin}

def test_compute_origin_correlations_happy(tmp_path: Path):
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # 3-day series:
    # twitter: [1,2,3], reddit: [2,4,6]  -> strong positive corr
    # rss_news: [3,2,1]                  -> negative corr vs twitter
    flags = []
    for i in range(3):
        day = now - timedelta(days=2 - i)
        flags += [_log(day, "twitter")] * (i + 1)
        flags += [_log(day, "reddit")] * (2 * (i + 1))
        flags += [_log(day, "rss_news")] * (3 - i)

    fpath = tmp_path / "flags.jsonl"; fpath.write_text("\n".join(_j(r) for r in flags))
    tpath = tmp_path / "triggers.jsonl"; tpath.write_text("")

    out = compute_origin_correlations(fpath, tpath, days=7, interval="day")
    pairs = {(p["a"], p["b"]): p["correlation"] for p in out["pairs"]}

    # Expect strong positive between twitter & reddit
    tr = pairs.get(("reddit", "twitter")) or pairs.get(("twitter", "reddit"))
    assert tr is not None and tr > 0.95

    # Expect negative between twitter & rss_news
    tx = pairs.get(("rss_news", "twitter")) or pairs.get(("twitter", "rss_news"))
    assert tx is not None and tx < -0.9

def test_compute_origin_correlations_empty(tmp_path: Path):
    fpath = tmp_path / "flags.jsonl"; fpath.write_text("")
    tpath = tmp_path / "triggers.jsonl"; tpath.write_text("")

    out = compute_origin_correlations(fpath, tpath, days=7, interval="day")
    assert out["pairs"] == []

def test_origin_correlations_api_demo_seed(tmp_path: Path, monkeypatch):
    # Point LOGS_DIR to empty tmp so the API sees no data
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path)
    (tmp_path / "retraining_log.jsonl").write_text("")
    (tmp_path / "retraining_triggered.jsonl").write_text("")

    old = os.environ.get("DEMO_MODE")
    os.environ["DEMO_MODE"] = "true"
    try:
        client = TestClient(app)
        res = client.get("/internal/origin-correlations?days=7&interval=day")
        assert res.status_code == 200
        data = res.json()
        assert data["pairs"] and len(data["pairs"]) >= 3
        for p in data["pairs"]:
            assert 0.3 <= p["correlation"] <= 0.8
    finally:
        if old is None:
            del os.environ["DEMO_MODE"]
        else:
            os.environ["DEMO_MODE"] = old
