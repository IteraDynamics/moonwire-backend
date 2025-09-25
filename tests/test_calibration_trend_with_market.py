# tests/test_calibration_trend_with_market.py
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.summary_sections.common import _iso
from scripts.summary_sections import calibration_reliability_trend as crt
import importlib

def _mk_hourly_series(start, hours, base, step):
    """Create a simple monotonic hourly price series."""
    out = []
    t = start
    price = base
    for _ in range(hours):
        out.append({"t": int(t.timestamp()), "price": float(price)})
        t += timedelta(hours=1)
        price += step
    return out

def test_enrich_json_with_market_and_alerts(tmp_path, monkeypatch):
    """
    Seed calibration_trend.json and a synthetic market_context.json with clear
    high-volatility late in the window. Ensure the enriched JSON includes:
      - market subobject with btc_return and btc_vol_bucket
      - alerts include both 'high_ece' and 'volatility_regime' when ECE > thresh AND vol is high
    Also verify plots are produced.
    """
    models = tmp_path / "models"; models.mkdir(parents=True, exist_ok=True)
    logs = tmp_path / "logs"; logs.mkdir(parents=True, exist_ok=True)
    arts = tmp_path / "artifacts"; arts.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    # Seed base calibration_trend.json (pre-enrichment shape)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [now - timedelta(hours=h) for h in (6, 4, 2)]
    trend = {
        "demo": False,
        "meta": {
            "demo": False,
            "dim": "origin",
            "window_h": 72,
            "bucket_min": 120,
            "ece_bins": 10,
            "generated_at": _iso(now),
        },
        "series": [
            {
                "key": "reddit",
                "points": [
                    {"bucket_start": _iso(buckets[0]), "ece": 0.04, "brier": 0.08, "n": 40},
                    {"bucket_start": _iso(buckets[1]), "ece": 0.05, "brier": 0.10, "n": 42},
                    # last bucket: high ECE to trigger 'high_ece'
                    {"bucket_start": _iso(buckets[2]), "ece": 0.12, "brier": 0.18, "n": 40},
                ],
            }
        ],
    }
    (models / "calibration_reliability_trend.json").write_text(json.dumps(trend))

    # Seed market_context.json with a spike in last 2–3 hours (raises volatility)
    start = now - timedelta(hours=72)
    btc_series = _mk_hourly_series(start, 72, 60000.0, 10.0)
    # Inject bigger moves in the last few hours to be above 75th percentile vol
    for k in range(1, 4):
        idx = -k
        btc_series[idx]["price"] += 500.0 * k

    # Build aligned ETH/SOL series using BTC timestamps
    eth_series = [{"t": btc_series[i]["t"], "price": 3000.0 + float(i) * 1.0} for i in range(len(btc_series))]
    sol_series = [{"t": btc_series[i]["t"], "price": 150.0 + float(i) * 0.1} for i in range(len(btc_series))]

    market = {
        "generated_at": _iso(now),
        "vs": "usd",
        "coins": ["bitcoin", "ethereum", "solana"],
        "window_hours": 72,
        "series": {
            "bitcoin": [{"t": p["t"], "price": p["price"]} for p in btc_series],
            "ethereum": eth_series,
            "solana": sol_series,
        },
        "returns": {},  # not required; code recomputes
        "demo": False,
        "attribution": "CoinGecko",
    }
    (models / "market_context.json").write_text(json.dumps(market))

    # Run enrichment
    importlib.reload(crt)
    ctx = type("Ctx", (), {})()
    ctx.logs_dir = logs
    ctx.models_dir = models
    ctx.artifacts_dir = arts
    ctx.is_demo = False

    md = []
    crt.append(md, ctx)

    # Assertions
    enriched = json.loads((models / "calibration_reliability_trend.json").read_text())
    assert enriched.get("series")
    # find our last reddit point
    reddit = next(s for s in enriched["series"] if s.get("key") == "reddit")
    last_pt = reddit["points"][-1]
    assert "market" in last_pt and "alerts" in last_pt
    assert "btc_return" in last_pt["market"]
    assert "btc_vol_bucket" in last_pt["market"]
    assert "high_ece" in last_pt["alerts"]
    assert "volatility_regime" in last_pt["alerts"]

    # Plots exist
    p1 = arts / "calibration_trend_ece.png"
    p2 = arts / "calibration_trend_brier.png"
    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0