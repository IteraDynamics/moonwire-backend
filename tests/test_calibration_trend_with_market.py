import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import os

from scripts.summary_sections.common import SummaryContext
from scripts.summary_sections import calibration_reliability_trend as crt


def _hourly_series(start_utc: datetime, hours: int):
    return [start_utc + timedelta(hours=i) for i in range(hours)]


def _epoch(ts: datetime) -> int:
    return int(ts.replace(tzinfo=timezone.utc).timestamp())


def test_enrich_json_with_market_and_alerts(tmp_path, monkeypatch):
    """
    Seed calibration_trend.json and a synthetic market_context.json with clear
    high-volatility late in the window. Ensure the enriched JSON includes:
      - market subobject with btc_return and btc_vol_bucket
      - alerts include both 'high_ece' and 'volatility_regime' when ECE > thresh AND vol is high
    Also verify plots are produced.
    """
    models = tmp_path / "models"; models.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    arts = tmp_path / "artifacts"; arts.mkdir()

    # Keep default ECE threshold or tighten if provided
    monkeypatch.setenv("MW_CAL_MAX_ECE", os.getenv("MW_CAL_MAX_ECE", "0.06"))

    # Build 24h hourly market data; last ~8h have large oscillating returns to trigger high vol
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    times = _hourly_series(now - timedelta(hours=23), 24)

    prices = []
    p = 100.0
    for i, t in enumerate(times):
        if i < 15:
            # near-flat regime
            p *= (1.0 + 0.0005)
        else:
            # volatile regime: alternate ±5%
            p *= (1.0 + (0.05 if i % 2 == 0 else -0.05))
        prices.append(p)

    market = {
        "generated_at": (now.isoformat().replace("+00:00", "Z")),
        "vs": "usd",
        "coins": ["bitcoin"],
        "series": {
            "bitcoin": [{"t": _epoch(t), "price": float(pr)} for t, pr in zip(times, prices)]
        },
        "demo": False,
        "attribution": "CoinGecko"
    }
    (models / "market_context.json").write_text(json.dumps(market))

    # Calibration buckets covering the last 10 hours; final bucket should align with high-vol span
    trend_rows = []
    for i in range(10):
        ts = (now - timedelta(hours=9 - i)).replace(minute=0, second=0, microsecond=0)
        # Mostly well-calibrated, last one bad
        ece = 0.12 if i == 9 else 0.03
        trend_rows.append({
            "bucket_start": ts.isoformat().replace("+00:00", "Z"),
            "ece": ece,
            "brier": 0.18 if i == 9 else 0.10,
            "n": 40 if i < 8 else 55,
            "label": "overall"
        })
    (models / "calibration_trend.json").write_text(json.dumps({"trend": trend_rows}))

    ctx = SummaryContext(logs_dir=logs, models_dir=models, is_demo=False)
    md = []
    crt.append(md, ctx)

    # Enriched JSON reloaded
    enriched = json.loads((models / "calibration_trend.json").read_text())
    assert isinstance(enriched, dict) and "trend" in enriched
    trend = enriched["trend"]
    assert len(trend) == 10

    last = trend[-1]
    # market fields present
    assert "market" in last and isinstance(last["market"], dict)
    assert "btc_return" in last["market"]
    assert last["market"]["btc_vol_bucket"] in ("high", "normal", None)
    # Because we forced heavy volatility late, expect high bucket on last item
    assert last["market"]["btc_vol_bucket"] == "high"

    # alerts carry both when high ECE during high-vol regime
    alerts = last.get("alerts", [])
    assert "high_ece" in alerts
    assert "volatility_regime" in alerts

    # Plots exist and are non-empty
    p_ece = Path(ctx.artifacts_dir) / "calibration_trend_ece.png"
    p_brier = Path(ctx.artifacts_dir) / "calibration_trend_brier.png"
    assert p_ece.exists() and p_ece.stat().st_size > 0
    assert p_brier.exists() and p_brier.stat().st_size > 0

    # Markdown produced at least one line
    assert any("Calibration & Reliability Trend vs Market Regimes" in line for line in md)


def test_demo_fallback_when_market_missing(tmp_path, monkeypatch):
    """
    Remove market_context.json and run in demo mode. The module should synthesize
    market regimes, enrich JSON, and still produce plots + markdown.
    """
    models = tmp_path / "models"; models.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    arts = tmp_path / "artifacts"; arts.mkdir()

    monkeypatch.setenv("MW_DEMO", "true")
    monkeypatch.setenv("MW_CAL_MAX_ECE", "0.06")

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Minimal calibration trend (3 buckets)
    trend_rows = []
    for i in range(3):
        ts = (now - timedelta(hours=2 - i)).replace(minute=0, second=0, microsecond=0)
        ece = 0.07 if i == 2 else 0.02
        trend_rows.append({
            "bucket_start": ts.isoformat().replace("+00:00", "Z"),
            "ece": ece,
            "brier": 0.12 + 0.01 * i,
            "n": 25,
            "label": "overall"
        })
    (models / "calibration_trend.json").write_text(json.dumps({"trend": trend_rows}))

    ctx = SummaryContext(logs_dir=logs, models_dir=models, is_demo=True)
    md = []
    crt.append(md, ctx)

    enriched = json.loads((models / "calibration_trend.json").read_text())
    trend = enriched["trend"]
    assert len(trend) == 3
    # Market subobject is present even in demo
    assert "market" in trend[-1]
    assert "btc_return" in trend[-1]["market"]
    assert trend[-1]["market"]["btc_vol_bucket"] in ("high", "normal", None)

    # Plots exist
    p_ece = Path(ctx.artifacts_dir) / "calibration_trend_ece.png"
    p_brier = Path(ctx.artifacts_dir) / "calibration_trend_brier.png"
    assert p_ece.exists() and p_ece.stat().st_size > 0
    assert p_brier.exists() and p_brier.stat().st_size > 0

    # Markdown contains demo note
    assert any("demo-seeded series" in line for line in md)


def test_volatility_percentile_logic(tmp_path, monkeypatch):
    """
    Construct a tiny market context where rolling vol is clearly higher in the
    last window; verify that at least one bucket lands in 'high' and earlier
    buckets are 'normal' or None.
    """
    models = tmp_path / "models"; models.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    arts = tmp_path / "artifacts"; arts.mkdir()

    monkeypatch.setenv("MW_CAL_MAX_ECE", "0.06")

    # 12 hours total, last 6 hours oscillate hard to boost vol
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    times = _hourly_series(now - timedelta(hours=11), 12)

    prices = []
    p = 200.0
    for i, t in enumerate(times):
        if i < 6:
            p *= (1.0 + 0.001)  # calm regime
        else:
            p *= (1.0 + (0.06 if i % 2 == 0 else -0.06))  # choppy regime
        prices.append(p)

    market = {
        "generated_at": (now.isoformat().replace("+00:00", "Z")),
        "vs": "usd",
        "coins": ["bitcoin"],
        "series": {
            "bitcoin": [{"t": _epoch(t), "price": float(pr)} for t, pr in zip(times, prices)]
        },
        "demo": False,
        "attribution": "CoinGecko"
    }
    (models / "market_context.json").write_text(json.dumps(market))

    # Trend aligned to last 6 hours
    trend_rows = []
    for i in range(6):
        ts = (now - timedelta(hours=5 - i)).replace(minute=0, second=0, microsecond=0)
        trend_rows.append({
            "bucket_start": ts.isoformat().replace("+00:00", "Z"),
            "ece": 0.04,
            "brier": 0.11,
            "n": 30,
            "label": "overall",
        })
    (models / "calibration_trend.json").write_text(json.dumps({"trend": trend_rows}))

    ctx = SummaryContext(logs_dir=logs, models_dir=models, is_demo=False)
    md = []
    crt.append(md, ctx)

    enriched = json.loads((models / "calibration_trend.json").read_text())
    trend = enriched["trend"]

    buckets = [row["market"]["btc_vol_bucket"] for row in trend]
    # Expect at least one 'high' in the later period
    assert "high" in buckets
    # And the early ones should be mostly normal/None
    early = buckets[:2]
    assert all(b in (None, "normal") for b in early)