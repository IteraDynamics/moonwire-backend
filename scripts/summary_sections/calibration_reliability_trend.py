# scripts/summary_sections/calibration_reliability_trend.py
"""
Calibration trend overlayed with market regimes (returns/volatility).

Inputs
------
- models/calibration_trend.json  (precomputed ECE/Brier trend buckets)
- models/market_context.json     (BTC/ETH/SOL hourly price series)

Outputs
-------
1) models/calibration_trend.json  (enriched in-place; adds `market` + `alerts`)
2) artifacts/calibration_trend_ece.png   (ECE over time with high-volatility bands)
3) artifacts/calibration_trend_brier.png (Brier over time with high-volatility bands)
4) Markdown lines appended via `append(md, ctx)`

Behavior
--------
- Computes hourly returns from BTC prices (falls back to demo-synth if missing).
- Rolling volatility proxy = std of last 6 hourly returns (per coin; we use BTC).
- Classifies volatility as 'high' when rolling vol >= 75th percentile over window.
- Aligns each calibration bucket (bucket_start, hour) to the BTC return/vol regime.
- Adds alerts:
    - 'high_ece' if ECE > threshold (env MW_CAL_MAX_ECE, default 0.06)
    - 'volatility_regime' if vol bucket == 'high'
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless safety
import matplotlib.pyplot as plt


# ------------------------------
# Small utilities
# ------------------------------

_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _parse_iso(s: str) -> datetime:
    # Accept both "...Z" and with offset; normalize to Z
    try:
        if s.endswith("Z"):
            return datetime.strptime(s, _ISO_FMT).replace(tzinfo=timezone.utc)
        # Fallback: fromisoformat then force UTC if missing
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        # very defensive: try fromisoformat raw
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(_ISO_FMT)


def _pct_change(prev: float, cur: float) -> Optional[float]:
    try:
        if prev is None or cur is None:
            return None
        if prev == 0:
            return None
        return (cur / prev) - 1.0
    except Exception:
        return None


def _rolling_std(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    buf: List[float] = []
    for v in values:
        if v is not None:
            buf.append(v)
        else:
            buf.append(float("nan"))
        if len(buf) > window:
            buf.pop(0)
        # compute std of non-nan only if we filled the window with finite values
        window_vals = [x for x in buf if (x is not None and not math.isnan(x))]
        if len(window_vals) < window:
            out.append(None)
        else:
            m = sum(window_vals) / len(window_vals)
            var = sum((x - m) ** 2 for x in window_vals) / len(window_vals)
            out.append(math.sqrt(var))
    return out


def _percentile(values: List[float], p: float) -> Optional[float]:
    vals = sorted([v for v in values if v is not None and not math.isnan(v)])
    if not vals:
        return None
    if p <= 0:
        return vals[0]
    if p >= 100:
        return vals[-1]
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return d0 + d1


# ------------------------------
# Data models
# ------------------------------

@dataclass
class SummaryContextLike:
    logs_dir: Path
    models_dir: Path
    artifacts_dir: Path
    is_demo: bool


# ------------------------------
# Market helpers
# ------------------------------

def _load_market_ctx(models_dir: Path) -> Optional[dict]:
    f = models_dir / "market_context.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def _synthesize_demo_market(now: Optional[datetime] = None, hours: int = 72) -> dict:
    """
    Build a small BTC series with a calm first half and choppy second half so
    tests can reliably observe a high-volatility regime.
    """
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    times = [now - timedelta(hours=(hours - 1 - i)) for i in range(hours)]
    prices = []
    p = 100.0
    for i in range(hours):
        if i < hours // 2:
            p *= (1.0 + 0.001)  # calm
        else:
            p *= (1.0 + (0.05 if i % 2 == 0 else -0.05))  # choppy
        prices.append(p)
    return {
        "generated_at": _iso(now),
        "vs": "usd",
        "coins": ["bitcoin"],
        "series": {
            "bitcoin": [{"t": int(ts.timestamp()), "price": float(pr)} for ts, pr in zip(times, prices)]
        },
        "demo": True,
        "attribution": "demo-seeded series",
    }


def _btc_times_prices(market_ctx: dict) -> Tuple[List[datetime], List[float]]:
    series = market_ctx.get("series", {})
    btc = series.get("bitcoin") or series.get("BTC") or []
    ts = []
    ps = []
    for row in btc:
        t_raw = row.get("t")
        pr = row.get("price")
        if t_raw is None or pr is None:
            continue
        # 't' may be seconds; treat as seconds
        try:
            t = datetime.fromtimestamp(int(t_raw), tz=timezone.utc)
        except Exception:
            # if given as ms, handle that
            t = datetime.fromtimestamp(int(t_raw) / 1000.0, tz=timezone.utc)
        ts.append(t.replace(minute=0, second=0, microsecond=0))
        ps.append(float(pr))
    # sort by time just in case
    zipped = sorted(zip(ts, ps), key=lambda x: x[0])
    if not zipped:
        return [], []
    ts_sorted, ps_sorted = list(zip(*zipped))
    return list(ts_sorted), list(ps_sorted)


def _returns_and_vol(ts: List[datetime], prices: List[float], vol_window: int = 6):
    if len(prices) < 2:
        return [None] * len(prices), [None] * len(prices)
    returns = [None]
    for i in range(1, len(prices)):
        returns.append(_pct_change(prices[i - 1], prices[i]))
    vol = _rolling_std(returns, window=vol_window)
    return returns, vol


def _vol_buckets(vol: List[Optional[float]], high_pct: float = 75.0) -> List[Optional[str]]:
    base = [v for v in vol if v is not None]
    thr = _percentile(base, high_pct) if base else None
    buckets: List[Optional[str]] = []
    for v in vol:
        if v is None or thr is None:
            buckets.append(None)
        else:
            buckets.append("high" if v >= thr else "normal")
    return buckets


# ------------------------------
# Calibration enrichment
# ------------------------------

def _load_trend(models_dir: Path) -> dict:
    f = models_dir / "calibration_trend.json"
    if not f.exists():
        return {"trend": []}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {"trend": []}


def _nearest_index(ts_list: List[datetime], target: datetime) -> Optional[int]:
    if not ts_list:
        return None
    # exact hour match first
    try:
        idx = ts_list.index(target)
        return idx
    except ValueError:
        pass
    # otherwise nearest in absolute time
    diffs = [(abs((t - target).total_seconds()), i) for i, t in enumerate(ts_list)]
    _, idx = min(diffs, key=lambda x: x[0])
    return idx


def _ece_thresh() -> float:
    try:
        return float(os.getenv("MW_CAL_MAX_ECE", "0.06"))
    except Exception:
        return 0.06


def _enrich_trend_with_market(trend: dict, market_ctx: dict) -> Tuple[dict, Dict[str, str]]:
    """
    Adds `market` and `alerts` to each trend row.

    Returns updated trend dict and a small summary for markdown.
    """
    times, prices = _btc_times_prices(market_ctx)
    returns, vol = _returns_and_vol(times, prices, vol_window=6)
    vol_bkts = _vol_buckets(vol, high_pct=75.0)

    # quick maps for lookups by time index
    idx_by_time = {t: i for i, t in enumerate(times)}

    ece_thr = _ece_thresh()

    hi_ece_and_hi_vol = 0
    total_hi_ece = 0

    for row in trend.get("trend", []):
        bstart = _parse_iso(row["bucket_start"])
        # align to hour to match our series normalization
        bstart = bstart.replace(minute=0, second=0, microsecond=0)

        idx = idx_by_time.get(bstart)
        if idx is None:
            idx = _nearest_index(times, bstart)

        r = returns[idx] if (idx is not None and 0 <= idx < len(returns)) else None
        vb = vol_bkts[idx] if (idx is not None and 0 <= idx < len(vol_bkts)) else None

        row_market = {
            "btc_return": r,
            "btc_vol_bucket": vb
        }
        row["market"] = row_market

        alerts: List[str] = row.get("alerts", [])
        # high_ece?
        if isinstance(row.get("ece"), (int, float)) and row["ece"] is not None:
            if row["ece"] > ece_thr:
                if "high_ece" not in alerts:
                    alerts.append("high_ece")
                total_hi_ece += 1

        # volatility_regime?
        if vb == "high":
            if "volatility_regime" not in alerts:
                alerts.append("volatility_regime")

        if "high_ece" in alerts and "volatility_regime" in alerts:
            hi_ece_and_hi_vol += 1

        row["alerts"] = alerts

    summary = {
        "high_ece_and_high_vol": str(hi_ece_and_hi_vol),
        "total_high_ece": str(total_hi_ece),
        "demo": "true" if market_ctx.get("demo") else "false",
        "market_attribution": str(market_ctx.get("attribution", "")),
    }
    return trend, summary


# ------------------------------
# Plotting
# ------------------------------

def _plot_with_vol_bands(
    times: List[datetime],
    vol_bkts: List[Optional[str]],
    x_points: List[datetime],
    y_series: List[Optional[float]],
    ylabel: str,
    out_path: Path,
) -> None:
    fig = plt.figure()
    ax = plt.gca()

    # Shade high-volatility hours
    # We merge contiguous 'high' regions into spans
    def bands():
        start = None
        for i, b in enumerate(vol_bkts):
            if b == "high" and start is None:
                start = i
            if (b != "high" or i == len(vol_bkts) - 1) and start is not None:
                end = i if b != "high" else i
                yield start, end
                start = None

    # convert index bands into time spans
    for s, e in bands():
        s = max(0, s)
        e = min(len(times) - 1, e)
        ax.axvspan(times[s], times[e], alpha=0.15, linewidth=0)

    # Plot the metric line
    xs = x_points
    ys = [float("nan") if v is None else v for v in y_series]
    ax.plot(xs, ys, marker="o", linewidth=1.5)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("time (UTC)")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _collect_series_for_plot(trend: dict) -> Tuple[List[datetime], List[Optional[float]], List[Optional[float]]]:
    xs: List[datetime] = []
    ece: List[Optional[float]] = []
    brier: List[Optional[float]] = []
    for row in trend.get("trend", []):
        xs.append(_parse_iso(row["bucket_start"]))
        ece.append(row.get("ece"))
        brier.append(row.get("brier"))
    return xs, ece, brier


# ------------------------------
# Public entry point used by tests
# ------------------------------

def append(md: List[str], ctx) -> None:
    """
    Enrich calibration trend with market regimes and emit plots + markdown.
    """
    # Support the SummaryContext used in tests
    models_dir: Path = Path(ctx.models_dir)
    artifacts_dir: Path = Path(ctx.artifacts_dir)
    is_demo: bool = bool(getattr(ctx, "is_demo", False))

    # Load/seed market context
    market_ctx = _load_market_ctx(models_dir)
    seeded_demo = False
    if market_ctx is None:
        if is_demo:
            market_ctx = _synthesize_demo_market()
            seeded_demo = True
        else:
            # No market; synthesize minimal so the section still works
            market_ctx = _synthesize_demo_market()
            seeded_demo = True

    # Load trend JSON
    trend = _load_trend(models_dir)

    # Enrich
    trend_enriched, summary = _enrich_trend_with_market(trend, market_ctx)

    # Persist enriched JSON (in-place update)
    (models_dir / "calibration_trend.json").write_text(json.dumps(trend_enriched))

    # Build plotting series
    times, prices = _btc_times_prices(market_ctx)
    returns, vol = _returns_and_vol(times, prices, vol_window=6)
    vol_bkts = _vol_buckets(vol, high_pct=75.0)

    xs, ece_vals, brier_vals = _collect_series_for_plot(trend_enriched)
    # normalize x to hour for plotting (matches shading times which are hourly)
    xs = [x.replace(minute=0, second=0, microsecond=0) for x in xs]

    # Write plots
    _plot_with_vol_bands(times, vol_bkts, xs, ece_vals, ylabel="ECE", out_path=artifacts_dir / "calibration_trend_ece.png")
    _plot_with_vol_bands(times, vol_bkts, xs, brier_vals, ylabel="Brier", out_path=artifacts_dir / "calibration_trend_brier.png")

    # Markdown
    md.append("### 🧮 Calibration & Reliability Trend vs Market Regimes (72h)")
    if seeded_demo:
        md.append("_Note: demo-seeded series used for market overlay (no market_context.json present)._")
    # Very small, readable summary
    hi_combo = summary.get("high_ece_and_high_vol", "0")
    hi_total = summary.get("total_high_ece", "0")
    attrib = summary.get("market_attribution", "")
    demo_flag = " (demo)" if summary.get("demo") == "true" else ""
    md.append(f"- High-ECE buckets overlapping high-volatility: **{hi_combo}/{hi_total}**")
    if attrib:
        md.append(f"- Market source: {attrib}{demo_flag}")