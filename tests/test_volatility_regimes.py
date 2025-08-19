import json, os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.analytics.volatility_regimes import compute_volatility_regimes
from src.analytics.threshold_policy import threshold_for_regime
from main import app
import src.paths as paths

def _j(d): return json.dumps(d, separators=(",", ":"))
def _log(ts, origin): return {"timestamp": ts.isoformat(), "origin": origin}

def _write_series(buf: list, start: datetime, origin: str, counts: list[int], step_hours=1):
    for i, c in enumerate(counts):
        ts = start + timedelta(hours=i * step_hours)
        for _ in range(int(c)):
            buf.append(_log(ts, origin))

def test_volatility_regimes_happy(tmp_path: Path):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=71)  # 72 buckets

    flags = []
    # twitter: high variance (spike)
    tw = [0]*60 + [10] + [0]*11
    # reddit: low variance
    rd = [1]*72
    # rss_news: moderate variance
    rs = [0]*30 + [2]*12 + [0]*30

    _write_series(flags, start, "twitter", tw)
    _write_series(flags, start, "reddit", rd)
    _write_series(flags, start, "rss_news", rs)

    fpath = tmp_path / "flags.jsonl"; fpath.write_text("\n".join(_j(r) for r in flags))
    tpath = tmp_path / "triggers.jsonl"; tpath.write_text("")

    out = compute_volatility_regimes(fpath, tpath, days=3, interval="hour", lookback=72, q_calm=0.33, q_turb=0.8)
    rows = {r["origin"]: r for r in out["origins"]}
    assert rows["twitter"]["vol_metric"] > rows["reddit"]["vol_metric"]
    assert rows["twitter"]["regime"] in ("turbulent", "normal")  # depending on quantile
    assert rows["reddit"]["regime"] in ("calm", "normal")

def test_quantile_edges_and_threshold_mapping():
    # mapping correctness
    assert threshold_for_regime("calm") == 2.2
    assert threshold_for_regime("normal") == 2.5
    assert threshold_for_regime("turbulent") == 3.0
    assert threshold_for_regime("other") == 2.5

def test_volatility_api_and_demo_seed(tmp_path: Path, monkeypatch):
    # point LOGS_DIR to empty tmp so APIs see no data
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path)
    (tmp_path / "retraining_log.jsonl").write_text("")
    (tmp_path / "retraining_triggered.jsonl").write_text("")

    old = os.environ.get("DEMO_MODE")
    os.environ["DEMO_MODE"] = "true"
    try:
        client = TestClient(app)
        res = client.get("/internal/volatility-regimes?days=30&interval=hour&lookback=72")
        assert res.status_code == 200
        data = res.json()
        assert data["origins"]  # seeded

        res2 = client.get("/internal/adaptive-thresholds?days=30&interval=hour")
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["origins"] and all("threshold" in o for o in data2["origins"])
    finally:
        if old is None: del os.environ["DEMO_MODE"]
        else: os.environ["DEMO_MODE"] = old

def test_param_validation():
    client = TestClient(app)
    # bad interval
    res = client.get("/internal/volatility-regimes?interval=minute")
    assert res.status_code == 422
    # bad lookback
    res = client.get("/internal/volatility-regimes?lookback=1")
    assert res.status_code == 422
