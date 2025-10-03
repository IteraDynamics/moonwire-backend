# scripts/summary_sections/cross_origin_analysis.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Any

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None  # We guard every pandas usage.

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # Also guarded.

from .common import SummaryContext


# -------------------------------
# Config helpers
# -------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


# -------------------------------
# Data loading / seeding
# -------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_jsonl(fp: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    return out


def _bucket_hourly_counts(rows: List[Dict[str, Any]], ts_key: str) -> Dict[str, int]:
    """
    Return dict: { 'YYYY-mm-ddTHH:00:00Z': count }
    """
    counts: Dict[str, int] = {}
    for r in rows:
        ts = r.get(ts_key) or r.get("ts") or r.get("created_utc") or r.get("timestamp")
        if not ts:
            continue
        try:
            if isinstance(ts, (int, float)):
                # seconds
                dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            dt = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
            key = dt.strftime("%Y-%m-%dT%H:00:00Z")
            counts[key] = counts.get(key, 0) + 1
        except Exception:
            continue
    return counts


def _bucket_market_returns(rows: List[Dict[str, Any]],
                           ts_key: str = "ts",
                           price_key: str = "price") -> Dict[str, float]:
    """
    Build hourly BTC returns (simple return) by hour.
    Rows should contain a timestamp and price; we group to hourly, take last price in hour,
    then compute r_t = (P_t / P_{t-1}) - 1.
    """
    if pd is None:
        return {}
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    # find columns heuristically
    for k in ["ts", "timestamp", ts_key]:
        if k in df.columns:
            ts_key = k
            break
    for k in ["price", "close", price_key]:
        if k in df.columns:
            price_key = k
            break

    if ts_key not in df.columns or price_key not in df.columns:
        return {}

    try:
        ts = pd.to_datetime(df[ts_key], utc=True, errors="coerce")
        df = df.assign(_ts=ts)
        df = df.dropna(subset=["_ts"])
        df["_hour"] = df["_ts"].dt.floor("H")
        # pick last price in each hour (could use mean as well)
        agg = df.sort_values("_ts").groupby("_hour")[price_key].last()
        ret = agg.pct_change().fillna(0.0)
        out: Dict[str, float] = {}
        for idx, val in ret.items():
            key = idx.strftime("%Y-%m-%dT%H:00:00Z")
            try:
                out[key] = float(val)
            except Exception:
                out[key] = 0.0
        return out
    except Exception:
        return {}


def _demo_series(window_h: int, seed: int = 7) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Create deterministic demo series for reddit_count, twitter_count, btc_return with
    plausible correlations/lead-lag:
      - reddit leads twitter by +1h
      - reddit leads market by +2h
      - twitter roughly synchronous with market
    """
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    idx = [now - timedelta(hours=h) for h in range(window_h)][::-1]

    # Base latent signal
    base = np.cumsum(rng.normal(0, 1, size=window_h)) + rng.normal(0, 0.5, size=window_h)

    # Reddit counts ~ positive drift of base + noise, then rectified to counts
    reddit_latent = base + rng.normal(0, 0.5, size=window_h)
    reddit = np.maximum(0, reddit_latent - np.min(reddit_latent) + 5.0)  # make positive
    # Normalize to reasonable count scale
    reddit = (reddit / np.max(reddit)) * 100.0 + 200.0

    # Twitter ~ reddit shifted by -1h (so reddit leads +1h)
    twitter = np.roll(reddit, +1) + rng.normal(0, 5.0, size=window_h)

    # BTC returns ~ reddit shifted by -2h (so reddit leads +2h) and scaled small
    market = np.roll(reddit, +2)
    market = (market - market.mean()) / (market.std() + 1e-6) * 0.01 + rng.normal(0, 0.002, size=window_h)

    def to_dict(series: np.ndarray, kind: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for t, v in zip(idx, series):
            key = t.strftime("%Y-%m-%dT%H:00:00Z")
            if kind == "count":
                out[key] = float(max(0.0, v))
            else:
                out[key] = float(v)
        return out

    return to_dict(reddit, "count"), to_dict(twitter, "count"), to_dict(market, "return")


# -------------------------------
# Series alignment & correlation
# -------------------------------

def _align_series(keys: List[str],
                  a: Dict[str, float],
                  b: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two dict time series on provided keys (ordered list of hour strings).
    Missing values → 0.0. Return arrays of equal length.
    """
    ax = np.array([float(a.get(k, 0.0) or 0.0) for k in keys], dtype=float)
    bx = np.array([float(b.get(k, 0.0) or 0.0) for k in keys], dtype=float)
    return ax, bx


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xv = x - np.mean(x)
    yv = y - np.mean(y)
    den = (np.linalg.norm(xv) * np.linalg.norm(yv))
    if den == 0:
        return float("nan")
    return float(np.dot(xv, yv) / den)


def _cross_corr_lag(x: np.ndarray, y: np.ndarray, max_shift: int) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """
    Compute cross-correlation for lags in [-max_shift, ..., +max_shift].
    We define positive lag L as: x leads y by L → compare x[:-L] with y[L:].
    Returns (best_lag, best_r, lags_array, r_values_array)
    """
    lags = np.arange(-max_shift, max_shift + 1, dtype=int)
    rvals: List[float] = []
    best_r = -np.inf
    best_lag = 0

    for L in lags:
        if L > 0:
            x_s = x[:-L] if L < len(x) else np.array([])
            y_s = y[L:] if L < len(y) else np.array([])
        elif L < 0:
            x_s = x[-L:] if -L < len(x) else np.array([])  # shift opposite
            y_s = y[:L] if -L < len(y) else np.array([])
        else:
            x_s = x
            y_s = y

        if len(x_s) < 3 or len(y_s) < 3 or len(x_s) != len(y_s):
            r = np.nan
        else:
            r = _pearson(x_s, y_s)
        rvals.append(r)

        if not np.isnan(r) and abs(r) > abs(best_r):
            best_r = r
            best_lag = int(L)

    return best_lag, float(best_r if not np.isnan(best_r) else np.nan), lags.astype(int), np.array(rvals, dtype=float)


def _perm_test_max_ccf(x: np.ndarray,
                       y: np.ndarray,
                       max_shift: int,
                       n_perm: int,
                       rng: np.random.Generator) -> float:
    """
    Permutation test for max |CCF|:
      - Compute observed best |r|
      - Shuffle y (permutation) n_perm times; compute best |r| each time
      - p = (count >= observed + 1) / (n_perm + 1)  (add-one smoothing)
    Note: Simple permutation breaks autocorrelation but is a pragmatic CI gate.
    """
    obs_lag, obs_r, _, _ = _cross_corr_lag(x, y, max_shift)
    obs = abs(obs_r) if not math.isnan(obs_r) else 0.0
    if n_perm <= 0 or obs == 0.0:
        return 1.0

    ge = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        _, r, _, _ = _cross_corr_lag(x, y_perm, max_shift)
        rv = abs(r) if not math.isnan(r) else 0.0
        if rv >= obs:
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return float(p)


# -------------------------------
# Public entry (section)
# -------------------------------

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build a full Lead–Lag Analysis section:
      - loads reddit/twitter counts & market returns
      - aligns to hourly buckets over a lookback window
      - computes pairwise Pearson (0-lag) and CCF with argmax lag
      - permutation test p-value for max |CCF|
      - writes JSON + PNG artifacts
      - renders CI markdown
    """
    lookback_h = _env_int("MW_LEADLAG_LOOKBACK_H", 72)
    max_shift_h = _env_int("MW_LEADLAG_MAX_SHIFT_H", 12)
    n_perm = _env_int("MW_LEADLAG_N_PERM", 100)
    demo = _env_bool("MW_DEMO", False)

    # Paths
    models_dir = "models"
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Load / seed series
    reddit_counts: Dict[str, float] = {}
    twitter_counts: Dict[str, float] = {}
    market_returns: Dict[str, float] = {}

    reddit_rows = _read_jsonl("logs/social_reddit.jsonl")
    twitter_rows = _read_jsonl("logs/social_twitter.jsonl")
    market_rows = _read_jsonl("logs/market_prices.jsonl")

    if demo or (not reddit_rows and not twitter_rows and not market_rows):
        # Seed deterministic demo series
        reddit_counts, twitter_counts, market_returns = _demo_series(lookback_h, seed=11)
        demo = True
    else:
        reddit_counts = _bucket_hourly_counts(reddit_rows, ts_key="created_utc")
        twitter_counts = _bucket_hourly_counts(twitter_rows, ts_key="created_utc")
        market_returns = _bucket_market_returns(market_rows, ts_key="ts", price_key="price")

    # Build the common hour index for the last lookback_h hours
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    keys = [(now - timedelta(hours=h)).strftime("%Y-%m-%dT%H:00:00Z") for h in range(lookback_h)][::-1]

    # Align arrays
    x_reddit, x_twitter = _align_series(keys, reddit_counts, twitter_counts)
    x_market = _align_series(keys, market_returns, market_returns)[0]  # a bit hacky; we only need one array

    # Pair maps
    pair_defs = [
        ("reddit–twitter", x_reddit, x_twitter),
        ("reddit–market", x_reddit, x_market),
        ("twitter–market", x_twitter, x_market),
    ]

    results: List[Dict[str, Any]] = []
    pearson_map: Dict[str, float] = {}

    rng = np.random.default_rng(1337)

    # Compute metrics per pair
    for name, a, b in pair_defs:
        # 0-lag Pearson
        r0 = _pearson(a, b)
        pearson_map[name.replace("–", "_").replace(" ", "")] = (None if math.isnan(r0) else float(r0))

        # Lead-lag via CCF
        lag, rbest, lags_arr, rvals = _cross_corr_lag(a, b, max_shift_h)

        # Permutation test for significance
        pval = _perm_test_max_ccf(a, b, max_shift_h, n_perm, rng) if not math.isnan(rbest) else 1.0
        sig = (pval < 0.05) if not math.isnan(rbest) else False

        results.append({
            "pair": name,
            "lag_hours": int(lag if not math.isnan(rbest) else 0),
            "r": None if math.isnan(rbest) else float(rbest),
            "p_value": float(pval),
            "significant": bool(sig),
            # For plotting convenience we persist the arrays (short series → fine)
            "_lags": lags_arr.tolist(),
            "_rvals": [None if math.isnan(v) else float(v) for v in rvals.tolist()],
        })

    # Write JSON
    out_json = {
        "window_hours": lookback_h,
        "max_shift_hours": max_shift_h,
        "generated_at": _utc_now_iso(),
        "pairs": results,
        "pearson_zero_lag": pearson_map,
        "demo": bool(demo),
    }
    with open(os.path.join(models_dir, "leadlag_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    # Plots
    if plt is not None:
        # Heatmap-like grid (3x3 but we only fill off-diagonals we have)
        try:
            fig = plt.figure(figsize=(4, 4), dpi=120)
            ax = fig.add_subplot(111)
            # Order: reddit, twitter, market
            # We’ll build a symmetric matrix with NaNs on diagonal
            labels = ["reddit", "twitter", "market"]
            M = np.full((3, 3), np.nan, dtype=float)
            # Fill from pearson_map (zero-lag)
            def get_r(a: str, b: str) -> float:
                k1 = f"{a}–{b}"
                k2 = f"{b}–{a}"
                v = None
                for k in (k1, k2):
                    kk = k.replace("–", "_").replace(" ", "")
                    if kk in pearson_map and pearson_map[kk] is not None:
                        v = pearson_map[kk]
                        break
                return float("nan") if v is None else float(v)

            M[0, 1] = get_r("reddit", "twitter")
            M[1, 0] = M[0, 1]
            M[0, 2] = get_r("reddit", "market")
            M[2, 0] = M[0, 2]
            M[1, 2] = get_r("twitter", "market")
            M[2, 1] = M[1, 2]

            # Show with imshow; NaNs render as white if we set colormap properly
            cax = ax.imshow(M, vmin=-1.0, vmax=1.0)
            ax.set_xticks([0, 1, 2], labels=labels)
            ax.set_yticks([0, 1, 2], labels=labels)
            for i in range(3):
                for j in range(3):
                    val = M[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Zero-lag Pearson (|r|≤1)")
            fig.tight_layout()
            fig.savefig(os.path.join(artifacts_dir, "leadlag_heatmap.png"))
            plt.close(fig)
        except Exception:
            pass

        # CCF plots per pair
        for rec in results:
            try:
                lags = rec.get("_lags") or []
                rvals = rec.get("_rvals") or []
                if not lags or not rvals:
                    continue
                fig = plt.figure(figsize=(5, 3), dpi=120)
                ax = fig.add_subplot(111)
                ax.plot(lags, rvals, lw=2)
                # Mark max
                if rec.get("r") is not None:
                    lag_star = int(rec.get("lag_hours") or 0)
                    ax.axvline(lag_star, linestyle="--")
                ax.set_xlabel("Lag (hours)  [positive: first origin leads]")
                ax.set_ylabel("CCF r")
                ax.set_title(f"CCF — {rec['pair']}")
                fig.tight_layout()
                fn = f"leadlag_ccf_{rec['pair'].replace('–','_').replace(' ','')}.png"
                fig.savefig(os.path.join(artifacts_dir, fn))
                plt.close(fig)
            except Exception:
                continue

    # CI markdown
    md.append(f"\n⏱️ Lead–Lag Analysis ({lookback_h}h, max ±{max_shift_h}h)")
    for rec in results:
        r = rec.get("r", None)
        if r is None or math.isnan(r):
            md.append(f"{rec['pair']:<18} → r=n/a | synchronous [p={rec['p_value']:.2f} {'✅' if rec['significant'] else '❌'}]")
            continue
        lag = int(rec.get("lag_hours") or 0)
        if lag == 0:
            note = "synchronous"
        elif lag > 0:
            note = f"{rec['pair'].split('–')[0]} leads by +{lag}h"
        else:
            # negative: second origin leads by +|lag|
            note = f"{rec['pair'].split('–')[1]} leads by +{abs(lag)}h"
        md.append(
            f"{rec['pair']:<18} → r={r:.2f} | {note} "
            f"[p={rec['p_value']:.2f} {'✅' if rec['significant'] else '❌'}]"
        )

    md.append("\n_Footer: Lead/lag via cross-correlation; significance via permutation test (p<0.05)._")
