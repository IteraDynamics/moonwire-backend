# scripts/summary_sections/calibration_reliability_trend.py
# v0.6.7 — Calibration trend overlaid with market regimes (returns / volatility)
#
# What this does:
# - Reads existing calibration trend JSON (models/calibration_trend.json)
# - Reads market context JSON (models/market_context.json)
# - Computes hourly returns and 6h rolling-vol per coin from market series
# - Tags each hour as "high-volatility" when rolling-vol > 75th percentile (72h window)
# - Enriches each calibration bucket with market stats (BTC return & vol bucket)
# - Adds alerts when (ECE > threshold) overlaps a high-volatility regime
# - Saves updated JSON + plots with shaded high-vol bands + markdown summary lines
#
# Safe in demo mode: falls back to seeded market regimes if context is missing.

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless in CI
import matplotlib.pyplot as plt  # noqa: E402

from .common import (
    SummaryContext,
    ensure_dir,
    _iso,
    _load_jsonl,
    _write_json,
)

# ----------------------------
# Tunables / env-driven knobs
# ----------------------------

# ECE threshold beyond which we consider "high_ece"
ECE_ALERT_THRESHOLD = float(os.getenv("MW_CAL_MAX_ECE", "0.06"))
# Percentile threshold for "high volatility" regime
HIGH_VOL_PERCENTILE = 0.75
# Rolling window (hours) for volatility proxy
ROLLING_VOL_WINDOW_H = 6
# Market coin we use for primary overlay fields (btc_return / btc_vol_bucket)
PRIMARY_MARKET_COIN = "bitcoin"  # BTC


# ----------------------------
# Helpers
# ----------------------------

def _parse_iso(ts: str) -> datetime:
    # Accepts "YYYY-MM-DDTHH:MMZ" or full iso; always return aware UTC
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        # robust fallback
        return datetime.strptime(ts.replace("Z", ""), "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)


def _pct_change(prev: float, cur: float) -> Optional[float]:
    try:
        if prev is None or cur is None or prev == 0:
            return None
        return (cur - prev) / prev
    except Exception:
        return None


def _rolling_std(values: List[Optional[float]], win: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    buf: List[float] = []
    for i, v in enumerate(values):
        if v is None:
            out.append(None)
            continue
        # grow buffer with non-None
        buf.append(v)
        # keep only last win non-None values
        j = i
        # Walk back to collect last 'win' non-None values
        acc: List[float] = []
        k = i
        while k >= 0 and len(acc) < win:
            vk = values[k]
            if vk is not None:
                acc.append(vk)
            k -= 1
        if len(acc) < max(2, min(win, i + 1)):  # need >=2 to compute std
            out.append(None)
            continue
        mean = sum(acc) / len(acc)
        var = sum((x - mean) ** 2 for x in acc) / (len(acc) - 1)
        out.append(math.sqrt(var))
    return out


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    idx = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
    return xs[idx]


@dataclass
class MarketSeries:
    times: List[datetime]           # hourly timestamps (UTC)
    prices: List[float]             # prices aligned to times
    returns: List[Optional[float]]  # hourly returns (t-1 -> t)
    vol6h: List[Optional[float]]    # rolling std of returns over 6h window
    high_vol_flags: List[bool]      # per-time label


@dataclass
class MarketContext:
    vs: str
    coins: List[str]
    series_by_coin: Dict[str, MarketSeries]
    generated_at: Optional[datetime]
    demo: bool


def _load_market_context(ctx: SummaryContext) -> Optional[MarketContext]:
    jpath = Path(ctx.models_dir) / "market_context.json"
    if not jpath.exists():
        return None
    try:
        raw = json.loads(jpath.read_text())
    except Exception:
        return None

    vs = raw.get("vs", "usd")
    coins = raw.get("coins", [])
    demo = bool(raw.get("demo", False))
    generated_at = None
    if "generated_at" in raw:
        try:
            generated_at = datetime.fromisoformat(raw["generated_at"].replace("Z", "+00:00"))
        except Exception:
            generated_at = None

    # Raw series are like {"bitcoin":[{"t": epoch_sec, "price": float}, ...], ...}
    series_field = raw.get("series", {})
    out: Dict[str, MarketSeries] = {}
    for coin in coins:
        rows = series_field.get(coin, [])
        if not rows:
            continue
        times = [datetime.fromtimestamp(r.get("t"), tz=timezone.utc) for r in rows if "t" in r]
        prices = [float(r.get("price")) for r in rows if "price" in r]
        # align lengths
        n = min(len(times), len(prices))
        times, prices = times[:n], prices[:n]

        # compute returns (hourly)
        rets: List[Optional[float]] = []
        prev = None
        for p in prices:
            rets.append(_pct_change(prev, p) if prev is not None else None)
            prev = p

        vol6 = _rolling_std(rets, ROLLING_VOL_WINDOW_H)

        # volatility percentile threshold over available window (ignore None)
        non_none_vols = [v for v in vol6 if v is not None]
        if non_none_vols:
            thr = _percentile(non_none_vols, HIGH_VOL_PERCENTILE)
        else:
            thr = float("inf")  # no data -> never high

        flags = [(v is not None and v > thr) for v in vol6]

        out[coin] = MarketSeries(
            times=times, prices=prices, returns=rets, vol6h=vol6, high_vol_flags=flags
        )

    if not out:
        return None
    return MarketContext(vs=vs, coins=list(out.keys()), series_by_coin=out, generated_at=generated_at, demo=demo)


def _synthesize_demo_market(ctx: SummaryContext) -> MarketContext:
    # Minimal, seeded 72h demo with mild volatility and a single high-vol spike near "now"
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    hours = list(range(72))
    times = [now - timedelta(hours=h) for h in reversed(hours)]
    base = {
        "bitcoin": 62000.0,
        "ethereum": 3100.0,
        "solana": 155.0,
    }
    series_by_coin: Dict[str, MarketSeries] = {}
    for coin, start in base.items():
        prices: List[float] = []
        p = start
        for i, t in enumerate(times):
            # tiny drift + single spike
            drift = 0.0005 * math.sin(i / 6.0)
            shock = -0.03 if (i == len(times) - 8) else 0.0  # one -3% shock ~ 8h ago
            p = max(1.0, p * (1.0 + drift + shock))
            prices.append(p)
        # returns
        rets: List[Optional[float]] = []
        prev = None
        for cur in prices:
            rets.append(_pct_change(prev, cur) if prev is not None else None)
            prev = cur
        vol6 = _rolling_std(rets, ROLLING_VOL_WINDOW_H)
        non_none_vols = [v for v in vol6 if v is not None]
        thr = _percentile(non_none_vols, HIGH_VOL_PERCENTILE) if non_none_vols else float("inf")
        flags = [(v is not None and v > thr) for v in vol6]
        series_by_coin[coin] = MarketSeries(times=times, prices=prices, returns=rets, vol6h=vol6, high_vol_flags=flags)

    return MarketContext(
        vs="usd",
        coins=list(series_by_coin.keys()),
        series_by_coin=series_by_coin,
        generated_at=now,
        demo=True,
    )


def _align_value_at(series_times: List[datetime], series_values: List[Optional[float]], target: datetime) -> Optional[float]:
    # Find value at exact hour; if not found, use nearest within ±30 minutes.
    # (Series is hourly, so exact match is expected. This is a safety valve.)
    if not series_times:
        return None
    # exact
    for t, v in zip(series_times, series_values):
        if t == target:
            return v
    # nearest
    best = None
    best_dt = timedelta(days=9999)
    for t, v in zip(series_times, series_values):
        dt = abs(t - target)
        if dt < best_dt and dt <= timedelta(minutes=30):
            best_dt = dt
            best = v
    return best


def _build_vol_spans(times: List[datetime], flags: List[bool]) -> List[Tuple[datetime, datetime]]:
    # Convert boolean flags into contiguous spans (start, end) for shading
    spans: List[Tuple[datetime, datetime]] = []
    if not times or not flags or len(times) != len(flags):
        return spans
    open_start: Optional[datetime] = None
    for i, (t, f) in enumerate(zip(times, flags)):
        if f and open_start is None:
            open_start = t
        if (not f) and open_start is not None:
            # close at current t
            spans.append((open_start, t))
            open_start = None
    # if ended high, close at last + 1h
    if open_start is not None:
        spans.append((open_start, times[-1] + timedelta(hours=1)))
    return spans


def _load_and_enrich_calibration(ctx: SummaryContext, market: MarketContext) -> Dict[str, List[dict]]:
    """
    Reads models/calibration_trend.json (expected top-level lists by metric or a flat list).
    We handle two shapes:
      A) {"trend": [ {bucket_start, ece, brier, n, ...}, ... ]}
      B) [ {bucket_start, ece, brier, n, ...}, ... ]   (flat list from earlier versions)

    Enrich each bucket with:
      "alerts": [...],
      "market": {"btc_return": float or null, "btc_vol_bucket": "high"|"normal"}
    """
    in_path = Path(ctx.models_dir) / "calibration_trend.json"
    if not in_path.exists():
        return {"trend": []}

    raw = json.loads(in_path.read_text())

    # Normalize to {"trend": [ ... ]}
    if isinstance(raw, list):
        trend = raw
        container = {"trend": trend}
    elif isinstance(raw, dict) and "trend" in raw and isinstance(raw["trend"], list):
        container = raw
        trend = raw["trend"]
    else:
        # Unknown shape – try best effort: collect dict/list items into "trend"
        trend = raw if isinstance(raw, list) else raw.get("trend", [])
        container = {"trend": trend}

    # Prepare BTC series for alignment
    btc = market.series_by_coin.get(PRIMARY_MARKET_COIN)
    btc_times: List[datetime] = btc.times if btc else []
    btc_returns: List[Optional[float]] = btc.returns if btc else []
    btc_high_flags: List[bool] = btc.high_vol_flags if btc else []

    # Enrich each bucket
    for row in trend:
        # bucket_start time
        ts_raw = row.get("bucket_start")
        if not ts_raw:
            row.setdefault("alerts", [])
            row.setdefault("market", {"btc_return": None, "btc_vol_bucket": None})
            continue
        t = _parse_iso(ts_raw)
        # Align to the market (hour resolution)
        btc_ret = _align_value_at(btc_times, btc_returns, t) if btc else None
        # For vol bucket, find exact index and check flag if available
        vol_bucket = None
        if btc:
            idx = None
            try:
                idx = btc_times.index(t)
            except ValueError:
                # try nearest within ±30 min
                nearest = None
                best_dt = timedelta(days=9999)
                for i, tt in enumerate(btc_times):
                    dt = abs(tt - t)
                    if dt < best_dt and dt <= timedelta(minutes=30):
                        best_dt = dt
                        nearest = i
                idx = nearest
            if idx is not None:
                is_high = bool(btc_high_flags[idx])
                vol_bucket = "high" if is_high else "normal"

        row_market = {
            "btc_return": float(btc_ret) if btc_ret is not None else None,
            "btc_vol_bucket": vol_bucket,
        }
        row["market"] = row_market

        # Alerts
        alerts: List[str] = list(row.get("alerts", []))
        ece = row.get("ece")
        if isinstance(ece, (int, float)) and float(ece) > ECE_ALERT_THRESHOLD:
            if vol_bucket == "high":
                # only add combined alert when both happen
                if "high_ece" not in alerts:
                    alerts.append("high_ece")
                if "volatility_regime" not in alerts:
                    alerts.append("volatility_regime")
            else:
                if "high_ece" not in alerts:
                    alerts.append("high_ece")
        else:
            # standalone volatility tag can still be informative
            if vol_bucket == "high" and "volatility_regime" not in alerts:
                alerts.append("volatility_regime")

        row["alerts"] = alerts

    # Write back enriched trend JSON
    out_path = Path(ctx.models_dir) / "calibration_trend.json"
    _write_json(out_path, container)
    return container


def _plot_with_overlays(ctx: SummaryContext, trend: List[dict], market: MarketContext) -> Tuple[Path, Path]:
    """
    Build two plots with shaded bands where BTC volatility is "high":
      - ECE vs time
      - Brier vs time
    Save to artifacts: calibration_trend_ece.png, calibration_trend_brier.png
    """
    ensure_dir(ctx.artifacts_dir)
    btc = market.series_by_coin.get(PRIMARY_MARKET_COIN)
    vol_spans = _build_vol_spans(btc.times, btc.high_vol_flags) if btc else []

    # Build calibration series
    times: List[datetime] = []
    ece_vals: List[float] = []
    brier_vals: List[float] = []

    for row in trend:
        ts = row.get("bucket_start")
        if not ts:
            continue
        t = _parse_iso(ts)
        times.append(t)
        ece_vals.append(float(row.get("ece")) if isinstance(row.get("ece"), (int, float)) else float("nan"))
        brier_vals.append(float(row.get("brier")) if isinstance(row.get("brier"), (int, float)) else float("nan"))

    # Plot ECE
    ece_path = Path(ctx.artifacts_dir) / "calibration_trend_ece.png"
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(times, ece_vals, marker="o", linewidth=1.5)
    # Shade high-vol spans (light grey)
    for (s, e) in vol_spans:
        ax1.axvspan(s, e, alpha=0.2, lw=0)
    ax1.set_title("Calibration Trend (ECE) with High-Volatility Bands")
    ax1.set_ylabel("ECE")
    ax1.set_xlabel("Time (UTC)")
    ax1.grid(True, alpha=0.3)
    fig1.autofmt_xdate()
    fig1.tight_layout()
    fig1.savefig(ece_path, dpi=150)
    plt.close(fig1)

    # Plot Brier
    brier_path = Path(ctx.artifacts_dir) / "calibration_trend_brier.png"
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(times, brier_vals, marker="o", linewidth=1.5)
    for (s, e) in vol_spans:
        ax2.axvspan(s, e, alpha=0.2, lw=0)
    ax2.set_title("Calibration Trend (Brier) with High-Volatility Bands")
    ax2.set_ylabel("Brier")
    ax2.set_xlabel("Time (UTC)")
    ax2.grid(True, alpha=0.3)
    fig2.autofmt_xdate()
    fig2.tight_layout()
    fig2.savefig(brier_path, dpi=150)
    plt.close(fig2)

    return ece_path, brier_path


def _summarize_for_markdown(trend: List[dict], market: MarketContext) -> List[str]:
    """
    Produce short, narrative lines like:
      reddit   → ECE ↑ during BTC −3% hour [volatility_regime]
      twitter  → stable calibration despite ETH −2%
      rss_news → noisy (low_n), high_ece not tied to market moves

    We don't have per-origin fields in this artifact here, so we summarize overall behavior:
    - find last bucket(s), report if high_ece coincided with volatility_regime and include btc_return
    """
    lines: List[str] = []
    if not trend:
        lines.append("No calibration trend data available.")
        return lines

    # Take last 3 buckets to summarize recent behavior
    recent = trend[-3:]

    def fmt_pct(x: Optional[float]) -> str:
        if x is None or not isinstance(x, (int, float)) or math.isnan(x):
            return "—"
        return f"{x:+.1%}"

    for row in recent:
        ece = row.get("ece")
        n = row.get("n")
        alerts = row.get("alerts", [])
        mkt = row.get("market", {}) or {}
        btc_ret = mkt.get("btc_return")
        vol = mkt.get("btc_vol_bucket")

        label = row.get("label") or "overall"
        parts = [f"{label:<8} → "]

        if isinstance(ece, (int, float)):
            parts.append(f"ECE {ece:.3f}")
        else:
            parts.append("ECE —")

        # BTC return context
        parts.append(f" during BTC {fmt_pct(btc_ret)}")

        tags: List[str] = []
        if "high_ece" in alerts:
            tags.append("high_ece")
        if vol == "high" or "volatility_regime" in alerts:
            tags.append("volatility_regime")

        if tags:
            parts.append(" [" + ", ".join(tags) + "]")

        # low sample indicator
        if isinstance(n, int) and n < 20:
            parts.append(" (low_n)")

        lines.append("".join(parts))

    return lines


# ----------------------------
# Public entry points
# ----------------------------

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Replaces the existing calibration trend summary block with version that includes
    market-regime overlays and enriched JSON.
    """
    # Load market (live or demo fallback)
    market = _load_market_context(ctx)
    if market is None:
        # Fall back to seeded demo regimes so plots still show overlays
        market = _synthesize_demo_market(ctx)

    # Enrich calibration trend with market linkage
    enriched = _load_and_enrich_calibration(ctx, market)
    trend: List[dict] = enriched.get("trend", [])

    # Plots with overlays
    ece_png, brier_png = _plot_with_overlays(ctx, trend, market)

    # Markdown header
    md.append("### 🧮 Calibration & Reliability Trend vs Market Regimes (72h)")
    # Narrative lines
    for line in _summarize_for_markdown(trend, market):
        md.append(line)

    # Footer note when demo market was used
    if market.demo:
        md.append("_Note: Market regime overlay uses demo-seeded series in this run._")


def main() -> None:
    """
    Manual/debug entry:
      python -m scripts.summary_sections.calibration_reliability_trend
    Writes markdown to stdout for inspection.
    """
    import sys
    # Try to infer repo root layout: ./logs, ./models, ./artifacts
    logs = Path(os.getenv("LOGS_DIR") or "./logs")
    models = Path(os.getenv("MODELS_DIR") or "./models")
    arts = Path(os.getenv("ARTIFACTS_DIR") or "./artifacts")
    ensure_dir(logs); ensure_dir(models); ensure_dir(arts)

    ctx = SummaryContext(logs_dir=logs, models_dir=models, is_demo=(os.getenv("MW_DEMO") == "true"))
    md: List[str] = []
    append(md, ctx)
    sys.stdout.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()