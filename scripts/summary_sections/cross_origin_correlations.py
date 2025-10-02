# scripts/summary_sections/cross_origin_correlation.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # CI-safe
import matplotlib.pyplot as plt

from .common import SummaryContext  # lightweight dependency only


# -------------------------
# Small utils (local-only)
# -------------------------
def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_ts(ts: Any) -> Optional[datetime]:
    # Supports ISO strings; safe fallback returns None
    if isinstance(ts, str):
        try:
            # Accept ...Z or with offset
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            return datetime.fromisoformat(ts).astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _floor_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _pair_key(a: str, b: str) -> str:
    return f"{a}_{b}" if a < b else f"{b}_{a}"


def _lead_lag_label(src: str, dst: str, lag_hours: int) -> str:
    # positive lag means `src` leads `dst`
    if lag_hours > 0:
        return f"{src}→{dst}", f"+{lag_hours}h"
    if lag_hours < 0:
        return f"{dst}→{src}", f"+{abs(lag_hours)}h"
    return f"{src}→{dst}", "0h"


@dataclass
class SeriesBundle:
    # aligned hourly time grid and values for reddit, twitter, market
    t: List[datetime]
    reddit: np.ndarray
    twitter: np.ndarray
    market: np.ndarray


# -------------------------
# Core computations
# -------------------------
def _aggregate_series(ctx: SummaryContext, lookback_h: int) -> SeriesBundle:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=lookback_h)

    # Build hourly grid (inclusive of start's hour, up to now's previous hour)
    t_grid: List[datetime] = []
    cur = _floor_hour(start)
    end = _floor_hour(now)
    while cur <= end:
        t_grid.append(cur)
        cur += timedelta(hours=1)

    # Reddit counts by hour
    reddit_log = Path(ctx.logs_dir or ".") / "social_reddit.jsonl"
    r_counts: Dict[datetime, int] = {}
    for r in _read_jsonl(reddit_log):
        ts = _parse_ts(r.get("created_utc") or r.get("timestamp") or r.get("ts"))
        if not ts:
            continue
        if ts < start or ts > now:
            continue
        h = _floor_hour(ts)
        r_counts[h] = r_counts.get(h, 0) + 1

    # Twitter counts by hour
    twitter_log = Path(ctx.logs_dir or ".") / "social_twitter.jsonl"
    tw_counts: Dict[datetime, int] = {}
    for r in _read_jsonl(twitter_log):
        ts = _parse_ts(r.get("created_utc") or r.get("timestamp") or r.get("ts"))
        if not ts:
            continue
        if ts < start or ts > now:
            continue
        h = _floor_hour(ts)
        tw_counts[h] = tw_counts.get(h, 0) + 1

    # Market: try logs/market_prices.jsonl  → compute 1h return on close
    # Fallback: models/market_context.json returns.h1 series if available
    market_log = Path(ctx.logs_dir or ".") / "market_prices.jsonl"
    mk_returns: Dict[datetime, float] = {}

    if market_log.exists():
        # Expect rows with {"t": <epoch or iso>, "symbol":"BTC", "price": float}
        # We'll take hourly close and compute returns
        hourly_price: Dict[datetime, float] = {}
        for r in _read_jsonl(market_log):
            if (r.get("symbol") or "").lower() not in ("btc", "bitcoin"):
                continue
            # Support epoch seconds or ISO
            t_raw = r.get("t") or r.get("ts") or r.get("timestamp")
            ts = None
            if isinstance(t_raw, (int, float)):
                try:
                    ts = datetime.fromtimestamp(float(t_raw), tz=timezone.utc)
                except Exception:
                    ts = None
            if ts is None:
                ts = _parse_ts(t_raw)
            if not ts:
                continue
            if ts < start or ts > now:
                continue
            h = _floor_hour(ts)
            hourly_price[h] = float(r.get("price", 0.0) or 0.0)

        # Sort and forward fill simple, then returns
        hh = sorted(hourly_price.keys())
        for i, h in enumerate(hh):
            if i == 0:
                continue
            p0 = hourly_price[hh[i - 1]]
            p1 = hourly_price[h]
            if p0 and p1:
                mk_returns[h] = (p1 - p0) / p0
    else:
        # Fallback to models/market_context.json if available
        mc_path = Path(ctx.models_dir or ".") / "market_context.json"
        if mc_path.exists():
            try:
                mc = json.loads(mc_path.read_text())
                # Try bitcoin series if present: {"series":{"bitcoin":[{"t": epoch, "price":...}, ...]}}
                series = ((mc.get("series") or {}).get("bitcoin")) or []
                hourly_price: Dict[datetime, float] = {}
                for row in series:
                    t_raw = row.get("t")
                    ts = None
                    if isinstance(t_raw, (int, float)):
                        try:
                            ts = datetime.fromtimestamp(float(t_raw), tz=timezone.utc)
                        except Exception:
                            ts = None
                    if not ts:
                        continue
                    if ts < start or ts > now:
                        continue
                    h = _floor_hour(ts)
                    hourly_price[h] = float(row.get("price", 0.0) or 0.0)
                hh = sorted(hourly_price.keys())
                for i, h in enumerate(hh):
                    if i == 0:
                        continue
                    p0 = hourly_price[hh[i - 1]]
                    p1 = hourly_price[h]
                    if p0 and p1:
                        mk_returns[h] = (p1 - p0) / p0
            except Exception:
                pass

    r_vec = np.array([r_counts.get(h, 0) for h in t_grid], dtype=float)
    t_vec = np.array([tw_counts.get(h, 0) for h in t_grid], dtype=float)
    m_vec = np.array([mk_returns.get(h, 0.0) for h in t_grid], dtype=float)

    return SeriesBundle(t=t_grid, reddit=r_vec, twitter=t_vec, market=m_vec)


def _pearsons(bundle: SeriesBundle) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pairs = [
        ("reddit", bundle.reddit, "twitter", bundle.twitter),
        ("reddit", bundle.reddit, "market", bundle.market),
        ("twitter", bundle.twitter, "market", bundle.market),
    ]
    for a_name, a, b_name, b in pairs:
        r = _safe_corr(a, b)
        out[_pair_key(a_name, b_name)] = 0.0 if math.isnan(r) else float(r)
    return out


def _max_xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    # Standardize (avoid NaN if constant)
    def _z(v):
        v = np.asarray(v, dtype=float)
        mu = v.mean() if v.size else 0.0
        sd = v.std() if v.size else 0.0
        if sd == 0:
            return v * 0.0
        return (v - mu) / (sd + 1e-9)

    az = _z(a)
    bz = _z(b)

    best_lag = 0
    best_corr = -1.0
    n = len(az)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # a lags (b leads): correlate a[t], b[t - lag]
            a_s = az[-lag:n]
            b_s = bz[0:n + lag]
        elif lag > 0:
            # a leads: correlate a[t - lag], b[t]
            a_s = az[0:n - lag]
            b_s = bz[lag:n]
        else:
            a_s = az
            b_s = bz
        if len(a_s) < 2 or len(b_s) < 2:
            continue
        r = _safe_corr(a_s, b_s)
        if r > best_corr:
            best_corr = r
            best_lag = lag
    return int(best_lag)


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# -------------------------
# Public entry: append()
# -------------------------
def append(md: List[str], ctx: SummaryContext) -> None:
    """Compute cross-origin Pearson and lead/lag; write JSON, PNGs; render markdown."""
    try:
        lookback_h = int(os.getenv("MW_CORR_LOOKBACK_H", "72"))
    except Exception:
        lookback_h = 72

    models_dir = Path(ctx.models_dir or "models")
    artifacts_dir = Path(getattr(ctx, "artifacts_dir", "artifacts"))
    _ensure_dir(models_dir / "cross_origin_correlation.json")  # ensure models dir at least

    demo = bool(getattr(ctx, "is_demo", False) or os.getenv("DEMO_MODE") == "true" or os.getenv("MW_DEMO") == "true")

    if demo:
        # Seeded deterministic outputs
        pearson = {
            "reddit_twitter": 0.65,
            "reddit_market": 0.35,
            "twitter_market": 0.40,
        }
        lead_lag_map = {
            "reddit→twitter": "+1h",
            "twitter→market": "0h",
            "reddit→market": "+2h",
        }
        # Create simple demo plots so assets exist
        _plot_heatmap(pearson, artifacts_dir / "corr_heatmap.png")
        _plot_leadlag({"reddit_twitter": 1, "twitter_market": 0, "reddit_market": 2}, artifacts_dir / "corr_leadlag.png")

        out = {
            "window_hours": lookback_h,
            "generated_at": _iso(datetime.now(timezone.utc)),
            "pearson": pearson,
            "lead_lag": lead_lag_map,
            "demo": True,
        }
        (models_dir / "cross_origin_correlation.json").write_text(json.dumps(out, indent=2))
        md.append(f"\n### 🔗 Cross-Origin Correlations ({lookback_h}h) (demo)")
        md.append(f"reddit–twitter   → r={pearson['reddit_twitter']:.2f} | reddit leads by ~1h")
        md.append(f"reddit–market    → r={pearson['reddit_market']:.2f} | reddit leads by ~2h")
        md.append(f"twitter–market   → r={pearson['twitter_market']:.2f} | synchronous")
        return

    # Live path
    bundle = _aggregate_series(ctx, lookback_h)

    pearsons = _pearsons(bundle)
    # Lead/lag across pairs (±6h)
    lag_cap = 6
    lag_map_int: Dict[str, int] = {
        "reddit_twitter": _max_xcorr_lag(bundle.reddit, bundle.twitter, lag_cap),
        "reddit_market": _max_xcorr_lag(bundle.reddit, bundle.market, lag_cap),
        "twitter_market": _max_xcorr_lag(bundle.twitter, bundle.market, lag_cap),
    }

    # Pretty lead-lag labels
    lead_lag_labels: Dict[str, str] = {}
    for key, lag in lag_map_int.items():
        a, b = key.split("_")
        _, label = _lead_lag_label(a, b, lag)
        lead_lag_labels[f"{a}→{b}"] = label

    # Persist JSON
    out = {
        "window_hours": lookback_h,
        "generated_at": _iso(datetime.now(timezone.utc)),
        "pearson": {
            "reddit_twitter": pearsons.get("reddit_twitter", 0.0),
            "reddit_market": pearsons.get("reddit_market", 0.0),
            "twitter_market": pearsons.get("twitter_market", 0.0),
        },
        "lead_lag": lead_lag_labels,
        "demo": False,
    }
    (models_dir / "cross_origin_correlation.json").write_text(json.dumps(out, indent=2))

    # Plots
    _plot_heatmap(out["pearson"], artifacts_dir / "corr_heatmap.png")
    _plot_leadlag(lag_map_int, artifacts_dir / "corr_leadlag.png")

    # Markdown
    md.append(f"\n### 🔗 Cross-Origin Correlations ({lookback_h}h)")
    md.append(f"reddit–twitter   → r={out['pearson']['reddit_twitter']:.2f} | "
              f"{_lead_phrase('reddit', 'twitter', lag_map_int['reddit_twitter'])}")
    md.append(f"reddit–market    → r={out['pearson']['reddit_market']:.2f} | "
              f"{_lead_phrase('reddit', 'market', lag_map_int['reddit_market'])}")
    md.append(f"twitter–market   → r={out['pearson']['twitter_market']:.2f} | "
              f"{_lead_phrase('twitter', 'market', lag_map_int['twitter_market'])}")


def _lead_phrase(a: str, b: str, lag: int) -> str:
    if lag > 0:
        return f"{a} leads by ~{lag}h"
    if lag < 0:
        return f"{b} leads by ~{abs(lag)}h"
    return "synchronous"


def _plot_heatmap(pearson: Dict[str, float], out_path: Path) -> None:
    names = ["reddit", "twitter", "market"]
    m = np.eye(3)
    # fill symmetric
    p_rt = float(pearson.get("reddit_twitter", 0.0) or 0.0)
    p_rm = float(pearson.get("reddit_market", 0.0) or 0.0)
    p_tm = float(pearson.get("twitter_market", 0.0) or 0.0)
    m[0, 1] = m[1, 0] = p_rt
    m[0, 2] = m[2, 0] = p_rm
    m[1, 2] = m[2, 1] = p_tm

    fig = plt.figure(figsize=(4.2, 3.6))
    ax = plt.gca()
    im = ax.imshow(m, vmin=-1, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Pearson correlation")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_leadlag(lags: Dict[str, int], out_path: Path) -> None:
    # map pair->lag; plot signed hours
    pairs = ["reddit_twitter", "reddit_market", "twitter_market"]
    vals = [int(lags.get(k, 0)) for k in pairs]

    fig = plt.figure(figsize=(4.8, 3.2))
    ax = plt.gca()
    ax.bar(range(len(pairs)), vals)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p.replace("_", "–") for p in pairs], rotation=0)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("Lead (hours)  — positive = left name leads")
    ax.set_title("Max cross-correlation lag (±6h)")
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
