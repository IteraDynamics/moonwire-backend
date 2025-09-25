# tests/test_calibration_trend_with_market.py
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.summary_sections.common import SummaryContext
from scripts.summary_sections.calibration_reliability_trend import append as crt_append
from scripts.summary_sections.calibration_reliability_trend import _iso

def _mk_hourly_series(start_ts: datetime, hours: int, start_price: float, step: float):
    pts = []
    p = start_price
    for i in range(hours):
        t = start_ts + timedelta(hours=i)
        pts.append({"t": int(t.timestamp()), "price": p})
        p += step
    return pts

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

    # Seed market_context.json with a spike in last 2 hours (raises volatility)
    start = now - timedelta(hours=72)
    btc_series = _mk_hourly_series(start, 72, 60000.0, 10.0)
    # Inject bigger moves in the last few hours to be above 75th percentile vol
    for k in range(1, 4):
        idx = -k
        btc_series[idx]["price"] += 500.0 * k

    market = {
        "generated_at": _iso(now),
        "vs": "usd",
        "coins": ["bitcoin", "ethereum", "solana"],
        "window_hours": 72,
        "series": {
            "bitcoin": [{"t": p["t"], "price": p["price"]} for p in btc_series],
            "ethereum": [{"t": p["t"], "price": 3000.0 + (i * 1.0)} for i, _ in enumerate(btc_series)],
            "solana": [{"t": p["t"], "price": 150.0 + (i * 0.1)} for i, _ in enumerate(btc_series)],
        },
        "returns": {},  # not required; code recomputes
        "demo": False,
        "attribution": "CoinGecko",
    }
    (models / "market_context.json").write_text(json.dumps(market))

    # Run enrichment + plotting via summary section
    ctx = SummaryContext(logs_dir=logs, models_dir=models, is_demo=False)
    md = []
    crt_append(md, ctx)

    # Markdown should mention the section header at least
    out = "\n".join(md)
    assert "Calibration & Reliability Trend" in out

    # Enriched JSON exists and has market fields/alerts
    enriched = json.loads((models / "calibration_reliability_trend.json").read_text())
    assert enriched.get("series")
    series = {s["key"]: s for s in enriched["series"]}
    assert "reddit" in series and len(series["reddit"]["points"]) >= 3

    last_pt = series["reddit"]["points"][-1]
    assert "market" in last_pt and "btc_return" in last_pt["market"]
    assert last_pt["market"].get("btc_vol_bucket") in ("low", "mid", "high")
    # With our seeded spike + ece=0.12, we expect both alerts
    assert set(last_pt.get("alerts", [])) >= {"high_ece", "volatility_regime"}

    # Plots exist
    assert (tmp_path / "artifacts" / "calibration_trend_ece.png").exists()
    assert (tmp_path / "artifacts" / "calibration_trend_brier.png").exists()