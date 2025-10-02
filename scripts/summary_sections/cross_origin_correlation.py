# scripts/summary_sections/cross_origin_correlation.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Numpy/matplotlib are already used elsewhere in the repo
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ensure headless
import matplotlib.pyplot as plt

from .common import SummaryContext, _iso


@dataclass
class SeriesBundle:
    # mapping from bucket_start (UTC, iso) -> float
    by_hour: Dict[str, float]
    label: str  # pretty label for plots


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _now_floor_hour() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _make_hour_grid(lookback_h: int) -> List[str]:
    now = _now_floor_hour()
    return [_iso(now - timedelta(hours=h)) for h in reversed(range(lookback_h))]


def _parse_ts(*vals: Optional[str]) -> Optional[datetime]:
    for v in vals:
        if not v:
            continue
        try:
            # Accept ISO or epoch seconds in string
            if v.isdigit():
                return datetime.fromtimestamp(int(v), tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return dt.replace(minute=0, second=0, microsecond=0)
        except Exception:
            continue
    return None


def _load_reddit_counts(ctx: SummaryContext, lookback_h: int) -> SeriesBundle:
    # Prefer append-only log (richest)
    by_hour: Dict[str, float] = {}
    log_path = (ctx.logs_dir or Path("logs")) / "social_reddit.jsonl"
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = _parse_ts(row.get("created_utc"), row.get("timestamp"), row.get("ts_ingested_utc"))
                if not ts:
                    continue
                key = _iso(ts)
                by_hour[key] = by_hour.get(key, 0.0) + 1.0

    # If empty, try to infer from context bursts (sparse, but better than nothing)
    if not by_hour:
        ctx_path = (ctx.models_dir or Path("models")) / "social_reddit_context.json"
        if ctx_path.exists():
            try:
                data = json.loads(ctx_path.read_text())
                for b in data.get("bursts", []) or []:
                    key = b.get("bucket_start")
                    if isinstance(b.get("posts"), (int, float)) and key:
                        by_hour[key] = max(by_hour.get(key, 0.0), float(b["posts"]))
            except Exception:
                pass

    # Restrict to window
    grid = _make_hour_grid(lookback_h)
    by_hour = {k: by_hour.get(k, 0.0) for k in grid}
    return SeriesBundle(by_hour=by_hour, label="Reddit posts")


def _load_twitter_counts(ctx: SummaryContext, lookback_h: int) -> SeriesBundle:
    by_hour: Dict[str, float] = {}
    log_path = (ctx.logs_dir or Path("logs")) / "social_twitter.jsonl"
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = _parse_ts(row.get("created_utc"), row.get("timestamp"), row.get("ts_ingested_utc"))
                if not ts:
                    continue
                key = _iso(ts)
                by_hour[key] = by_hour.get(key, 0.0) + 1.0

    # Fallback: derive sparse counts from context bursts if available
    if not by_hour:
        ctx_path = (ctx.models_dir or Path("models")) / "social_twitter_context.json"
        if ctx_path.exists():
            try:
                data = json.loads(ctx_path.read_text())
                for b in data.get("bursts", []) or []:
                    key = b.get("bucket_start")
                    if isinstance(b.get("tweets"), (int, float)) and key:
                        by_hour[key] = max(by_hour.get(key, 0.0), float(b["tweets"]))
            except Exception:
                pass

    grid = _make_hour_grid(lookback_h)
    by_hour = {k: by_hour.get(k, 0.0) for k in grid}
    return SeriesBundle(by_hour=by_hour, label="Twitter tweets")


def _load_market_returns(ctx: SummaryContext, lookback_h: int) -> SeriesBundle:
    """
    Try logs/market_prices.jsonl (hourly price). If missing, derive BTC hourly prices from
    models/market_context.json (the live CoinGecko section writes a dense series).
    Then compute 1h simple returns.
    """
    prices: Dict[str, float] = {}

    # 1) logs path (if someone wrote it previously)
    log_path = (ctx.logs_dir or Path("logs")) / "market_prices.jsonl"
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = _parse_ts(row.get("t_iso"), row.get("timestamp"))
                price = row.get("price")
                if ts and isinstance(price, (int, float)):
                    prices[_iso(ts)] = float(price)

    # 2) fallback: models/market_context.json (CoinGecko artifact)
    if not prices:
        ctx_path = (ctx.models_dir or Path("models")) / "market_context.json"
        if ctx_path.exists():
            try:
                data = json.loads(ctx_path.read_text())
                # coin series looks like { "series": { "bitcoin": [ {"t": epoch, "price": ...}, ... ] } }
                series = ((data.get("series") or {}).get("bitcoin")) or []
                for p in series:
                    # t can be epoch seconds; convert then floor to hour iso
                    ts = None
                    if "t" in p and isinstance(p["t"], (int, float)):
                        ts = datetime.fromtimestamp(int(p["t"]), tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
                    if not ts and "timestamp" in p:
                        ts = _parse_ts(p.get("timestamp"))
                    price = p.get("price")
                    if ts and isinstance(price, (int, float)):
                        prices[_iso(ts)] = float(price)
            except Exception:
                pass

    # Build returns over the window
    grid = _make_hour_grid(lookback_h)
    # sort by time, then compute simple 1h return
    vals: List[Tuple[str, float]] = [(k, prices.get(k)) for k in grid if k in prices]
    # Need consecutive hours; compute returns only where both t and t-1 exist
    ret_by_hour: Dict[str, float] = {}
    if vals:
        # Create index by key for contiguous detection
        grid_idx = {k: i for i, k in enumerate(grid)}
        for k in grid:
            prev_idx = grid_idx[k] - 1
            if prev_idx < 0:
                ret_by_hour[k] = np.nan
                continue
            k_prev = grid[prev_idx]
            p_t = prices.get(k)
            p_prev = prices.get(k_prev)
            if isinstance(p_t, (int, float)) and isinstance(p_prev, (int, float)) and p_prev != 0:
                ret_by_hour[k] = (p_t - p_prev) / p_prev
            else:
                ret_by_hour[k] = np.nan
    else:
        # fill NaN if no prices
        ret_by_hour = {k: np.nan for k in grid}

    # Replace NaNs with 0 for correlation safety only if we have at least some valid points;
    # we will later mask to overlap and drop all-zeros variance.
    ret_by_hour = {k: (0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)) for k, v in ret_by_hour.items()}
    return SeriesBundle(by_hour=ret_by_hour, label="BTC 1h returns")


def _align_vectors(a: SeriesBundle, b: SeriesBundle, grid: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    va = np.array([a.by_hour.get(k, 0.0) for k in grid], dtype=float)
    vb = np.array([b.by_hour.get(k, 0.0) for k in grid], dtype=float)
    return va, vb


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    # mask where both non-NaN (we already avoid NaN, but keep for safety)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return None
    # if either has zero variance, correlation undefined
    if np.allclose(x, x.mean()) or np.allclose(y, y.mean()):
        return None
    try:
        r = np.corrcoef(x, y)[0, 1]
        if np.isnan(r):
            return None
        return float(r)
    except Exception:
        return None


def _lead_lag_hours(x: np.ndarray, y: np.ndarray, max_lag: int = 6) -> Optional[int]:
    """
    Return lag in hours where x→y lead-lag correlation (cross-corr) is maximal.
    Positive lag means x **leads** y by `lag` hours.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 4:
        return None
    # z-score (avoid division by zero)
    def z(v):
        if np.std(v) == 0:
            return v * 0.0
        return (v - np.mean(v)) / np.std(v)
    xz, yz = z(x), z(y)
    best_lag = 0
    best_val = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # x lags y (y leads)
            xs = xz[-lag:]
            ys = yz[:len(xs)]
        elif lag > 0:
            # x leads y
            xs = xz[:len(xz) - lag]
            ys = yz[lag:]
        else:
            xs = xz
            ys = yz
        if len(xs) < 4 or len(ys) < 4:
            continue
        try:
            val = float(np.corrcoef(xs, ys)[0, 1])
        except Exception:
            continue
        if np.isnan(val):
            continue
        # choose by absolute value; if tie, prefer smaller |lag|
        score = abs(val)
        if (score > best_val) or (np.isclose(score, best_val) and abs(lag) < abs(best_lag)):
            best_val = score
            best_lag = lag
    return best_lag


def _save_heatmap(path: Path, labels: List[str], matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(matrix, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Pearson r")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_leadlag(path: Path, pair_labels: List[str], lags: List[Optional[int]]) -> None:
    # Convert lags to numeric with NaN->0 for plotting, but annotate with text
    vals = [0 if l is None else l for l in lags]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)), labels=pair_labels, rotation=45, ha="right")
    ax.set_ylabel("Lag (hours)  —  positive means left leads right")
    ax.set_title("Lead–Lag (max abs cross-correlation)")
    # annotate exact values
    for i, l in enumerate(lags):
        txt = "n/a" if l is None else f"{l:+d}h"
        ax.text(i, vals[i] + (0.3 if vals[i] >= 0 else -0.6), txt, ha="center", va="bottom" if vals[i]>=0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def append(md: List[str], ctx: SummaryContext) -> None:
    lookback_h = _env_int("MW_CORR_LOOKBACK_H", 72)
    demo_mode = str(os.getenv("MW_DEMO", os.getenv("DEMO_MODE", "false"))).lower() == "true"

    models_dir = ctx.models_dir or Path("models")
    arts_dir = ctx.artifacts_dir or Path("artifacts")
    arts_dir.mkdir(parents=True, exist_ok=True)

    # Build three series
    s_reddit = _load_reddit_counts(ctx, lookback_h)
    s_twitter = _load_twitter_counts(ctx, lookback_h)
    s_market = _load_market_returns(ctx, lookback_h)

    grid = _make_hour_grid(lookback_h)
    v_r, v_t = _align_vectors(s_reddit, s_twitter, grid)
    v_rm = _align_vectors(s_reddit, s_market, grid)
    v_tm = _align_vectors(s_twitter, s_market, grid)

    r_rt = _pearson_safe(v_r, v_t)
    r_rm = _pearson_safe(v_rm[0], v_rm[1])
    r_tm = _pearson_safe(v_tm[0], v_tm[1])

    lag_rt = _lead_lag_hours(v_r, v_t)
    lag_tm = _lead_lag_hours(v_t, v_tm[1])  # twitter vs market
    lag_rm = _lead_lag_hours(v_r, v_rm[1])  # reddit vs market

    # If demo and we failed to compute, seed plausible values
    if demo_mode and (r_rt is None or r_rm is None or r_tm is None):
        r_rt = 0.65 if r_rt is None else r_rt
        r_rm = 0.35 if r_rm is None else r_rm
        r_tm = 0.40 if r_tm is None else r_tm
        if lag_rt is None: lag_rt = 1
        if lag_rm is None: lag_rm = 2
        if lag_tm is None: lag_tm = 0

    # Assemble JSON artifact
    def _fmt_r(v: Optional[float]) -> Optional[float]:
        return None if v is None else float(np.clip(v, -1.0, 1.0))

    out = {
        "window_hours": lookback_h,
        "generated_at": _iso(_now_floor_hour()),
        "pearson": {
            "reddit_twitter": _fmt_r(r_rt),
            "reddit_market": _fmt_r(r_rm),
            "twitter_market": _fmt_r(r_tm),
        },
        "lead_lag": {
            "reddit→twitter": (f"{lag_rt:+d}h" if lag_rt is not None else None),
            "twitter→market": (f"{lag_tm:+d}h" if lag_tm is not None else None),
            "reddit→market": (f"{lag_rm:+d}h" if lag_rm is not None else None),
        },
        "demo": bool(demo_mode),
    }

    # Save JSON
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "cross_origin_correlation.json").write_text(json.dumps(out, indent=2))

    # Heatmap (pearson)
    labels = ["Reddit", "Twitter", "Market"]
    mat = np.full((3, 3), np.nan)
    # diagonal = 1
    for i in range(3):
        mat[i, i] = 1.0
    # off-diagonals
    if r_rt is not None:
        mat[0, 1] = mat[1, 0] = r_rt
    if r_rm is not None:
        mat[0, 2] = mat[2, 0] = r_rm
    if r_tm is not None:
        mat[1, 2] = mat[2, 1] = r_tm

    try:
        _save_heatmap(arts_dir / "corr_heatmap.png", labels, np.nan_to_num(mat, nan=0.0))
    except Exception:
        pass

    # Lead-lag bars
    pair_labels = ["reddit→twitter", "twitter→market", "reddit→market"]
    try:
        _save_leadlag(arts_dir / "corr_leadlag.png", pair_labels, [lag_rt, lag_tm, lag_rm])
    except Exception:
        pass

    # Markdown block
    def _fmt_line(name: str, r: Optional[float], lag: Optional[int]) -> str:
        rtxt = "n/a" if r is None else f"{r:.2f}"
        ltxt = "synchronous" if (lag == 0) else ("n/a" if lag is None else f"{lag:+d}h")
        return f"{name} → r={rtxt} | {('reddit' if 'reddit' in name else 'twitter' if 'twitter' in name.split('–')[0].lower() else name)} leads by {ltxt}" if ltxt not in ("synchronous", "n/a") else f"{name} → r={rtxt} | {ltxt}"

    md.append(f"\n### 🔗 Cross-Origin Correlations ({lookback_h}h)")
    md.append(_fmt_line("reddit–twitter", r_rt, lag_rt))
    md.append(_fmt_line("reddit–market", r_rm, lag_rm))
    md.append(_fmt_line("twitter–market", r_tm, lag_tm))
