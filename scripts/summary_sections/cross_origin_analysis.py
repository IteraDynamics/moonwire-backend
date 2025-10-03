# scripts/summary_sections/cross_origin_analysis.py
from __future__ import annotations

"""
Cross-Origin Analysis (v0.7.5)
- Computes pairwise Pearson correlations and robust lead–lag via cross-correlation.
- Adds permutation-test significance.
- Emits JSON + PNG artifacts and a CI-ready markdown block.

Artifacts:
- models/leadlag_analysis.json
- artifacts/leadlag_heatmap.png
- artifacts/leadlag_ccf_<pair>.png

Env knobs:
- MW_LEADLAG_LOOKBACK_H    (default: 72)
- MW_LEADLAG_MAX_SHIFT_H   (default: 12)
- MW_LEADLAG_N_PERM        (default: 100)
- DEMO_MODE                ("true"/"1" enables seeded outputs)

Inputs (append-only logs):
- logs/social_reddit.jsonl   (one line per post; we aggregate hourly counts)
- logs/social_twitter.jsonl  (one line per tweet; aggregate hourly counts)
- logs/market_prices.jsonl   (expects {ts, price} or {ts, returns_1h}; builds 1h returns if needed)
"""

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Matplotlib is only used to dump static PNGs (Agg backend is enforced in CI)
import matplotlib
matplotlib.use("Agg")  # safety: headless on CI
import matplotlib.pyplot as plt  # noqa: E402

# Local types/context
try:
    from .common import SummaryContext
except Exception:
    # Minimal shim so local/manual runs don't explode
    @dataclass
    class SummaryContext:  # type: ignore
        caches: Dict[str, Any] | None = None


# -------------------------
# Utilities
# -------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except Exception:
        return default

def _parse_ts(ts: str) -> Optional[datetime]:
    # Accept ISO-like stamps: "2025-10-01T12:34:56Z" or with offset
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def _truncate_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return rows

def _bucket_hourly_counts(rows: List[Dict[str, Any]], ts_field_candidates: Tuple[str, ...]) -> Dict[datetime, int]:
    counts: Dict[datetime, int] = {}
    for r in rows:
        ts = None
        for k in ts_field_candidates:
            if k in r and isinstance(r[k], str):
                ts = _parse_ts(r[k])
                if ts:
                    break
        if not ts:
            continue
        dt = _truncate_to_hour(ts.astimezone(timezone.utc))
        counts[dt] = counts.get(dt, 0) + 1
    return counts

def _align_series(hour_grid: List[datetime], series_map: Dict[datetime, float]) -> np.ndarray:
    # Return values aligned with hour_grid; fill missing with 0
    return np.array([float(series_map.get(h, 0.0)) for h in hour_grid], dtype=float)

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    sa = a.std()
    sb = b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def _hour_grid(lookback_h: int, now_utc: Optional[datetime] = None) -> List[datetime]:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    end = _truncate_to_hour(now_utc)
    start = end - timedelta(hours=lookback_h - 1)
    return [_truncate_to_hour(start + timedelta(hours=i)) for i in range(lookback_h)]

def _build_market_returns(
    rows: List[Dict[str, Any]],
    hour_grid: List[datetime],
) -> Dict[datetime, float]:
    """
    Accept either {ts, price} series or {ts, returns_1h}.
    Build hourly returns if price is present; prefer explicit returns_1h when available.
    """
    # First try explicit returns
    ret_map: Dict[datetime, float] = {}
    had_returns = False
    for r in rows:
        ts = _parse_ts(str(r.get("ts") or r.get("timestamp") or r.get("created_utc") or ""))
        if not ts:
            continue
        dt = _truncate_to_hour(ts.astimezone(timezone.utc))
        if "returns_1h" in r:
            try:
                ret_map[dt] = float(r["returns_1h"])
                had_returns = True
            except Exception:
                pass
    if had_returns:
        return ret_map

    # Else build from price
    price_map: Dict[datetime, float] = {}
    for r in rows:
        ts = _parse_ts(str(r.get("ts") or r.get("timestamp") or r.get("created_utc") or ""))
        if not ts:
            continue
        dt = _truncate_to_hour(ts.astimezone(timezone.utc))
        p = r.get("price") or r.get("close") or r.get("last") or r.get("value")
        try:
            price_map[dt] = float(p)
        except Exception:
            continue

    # Compute 1h simple returns
    prev = None
    for h in sorted(price_map.keys()):
        if prev is not None and price_map.get(prev, None) not in (None,) and price_map.get(h, None) not in (None,):
            p0 = price_map[prev]
            p1 = price_map[h]
            if p0 != 0:
                ret_map[h] = (p1 - p0) / p0
        prev = h
    return ret_map

# -------------------------
# Core cross-correlation (used by tests)
# -------------------------

def _cross_corr_lag(x: np.ndarray, y: np.ndarray, max_shift: int):
    """
    Cross-correlation over integer lags with a clear sign convention.

    Positive lag  L > 0  =>  x leads y by +L (compare x[0:n-L] with y[L:n]).
    Negative lag  L < 0  =>  y leads x by +|L| (compare x[|L|:n] with y[0:n-|L|]).

    Returns: (best_lag, r_best, lags_array, r_by_lag_array)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(min(len(x), len(y)))
    if n < 3:
        return 0, 0.0, np.arange(-max_shift, max_shift + 1), np.full(2 * max_shift + 1, np.nan)

    x = x[:n]
    y = y[:n]

    # Z-score (robust to scale/mean)
    xz = (x - x.mean()) / (x.std() + 1e-12)
    yz = (y - y.mean()) / (y.std() + 1e-12)

    lags = np.arange(-max_shift, max_shift + 1, dtype=int)
    r_vals = np.full_like(lags, fill_value=np.nan, dtype=float)

    for i, L in enumerate(lags):
        if L > 0:
            a = xz[: n - L]
            b = yz[L:]
        elif L < 0:
            k = -L
            a = xz[k:]
            b = yz[: n - k]
        else:
            a = xz
            b = yz

        if a.size >= 2 and b.size == a.size:
            r = np.corrcoef(a, b)[0, 1]
            r_vals[i] = float(r)

    if np.all(np.isnan(r_vals)):
        return 0, 0.0, lags, r_vals

    # Select lag with maximum |r|
    idx = int(np.nanargmax(np.abs(r_vals)))
    best_lag = int(lags[idx])
    r_best = float(r_vals[idx])
    return best_lag, r_best, lags, r_vals

def _perm_p_value(x: np.ndarray, y: np.ndarray, lag: int, r_obs: float, n_perm: int = 100, rng_seed: int = 0) -> float:
    """
    Permutation test on the observed lag by shuffling y (keeping x fixed).
    Compute correlation at the chosen lag for each permutation and return
    a two-sided p-value based on |r|.
    """
    rng = np.random.default_rng(rng_seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3 or not np.isfinite(r_obs):
        return 1.0

    # Build aligned slices for the observed lag
    if lag > 0:
        a_obs = x[: n - lag]
        b_obs = y[lag:]
    elif lag < 0:
        k = -lag
        a_obs = x[k:]
        b_obs = y[: n - k]
    else:
        a_obs = x
        b_obs = y

    # Z-score them once to mirror _cross_corr_lag
    def z(v):
        v = (v - v.mean()) / (v.std() + 1e-12)
        return v

    a_obs = z(a_obs)
    b_obs = z(b_obs)

    # Observed |r|
    r0 = float(np.corrcoef(a_obs, b_obs)[0, 1])
    r0_abs = abs(r0) if np.isfinite(r0) else 0.0

    # Permute y and compute r at same lag window
    count = 0
    for _ in range(int(n_perm)):
        y_perm = rng.permutation(y)
        if lag > 0:
            bp = y_perm[lag:]
            ap = x[: n - lag]
        elif lag < 0:
            k = -lag
            bp = y_perm[: n - k]
            ap = x[k:]
        else:
            ap = x
            bp = y_perm

        ap = z(ap)
        bp = z(bp)
        rp = float(np.corrcoef(ap, bp)[0, 1])
        if abs(rp) >= r0_abs:
            count += 1

    # +1 smoothing for small permutations
    pval = (count + 1) / (n_perm + 1)
    return float(min(max(pval, 0.0), 1.0))

# -------------------------
# Public API: append(md, ctx)
# -------------------------

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Compute lead/lag across origins and append a markdown block.
    Writes JSON + PNG artifacts as side effects.
    """
    # Env knobs
    lookback_h = _env_int("MW_LEADLAG_LOOKBACK_H", 72)
    max_shift_h = _env_int("MW_LEADLAG_MAX_SHIFT_H", 12)
    n_perm = _env_int("MW_LEADLAG_N_PERM", 100)
    demo = _env_flag("DEMO_MODE", False)

    # File paths
    reddit_log = "logs/social_reddit.jsonl"
    twitter_log = "logs/social_twitter.jsonl"
    market_log = "logs/market_prices.jsonl"
    out_json = "models/leadlag_analysis.json"
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # Build hour grid
    grid = _hour_grid(lookback_h)

    # -------------------------
    # Load data or seed demo
    # -------------------------
    series: Dict[str, np.ndarray] = {}
    labels: List[str] = ["reddit", "twitter", "market"]
    have_real_data = True

    if demo:
        have_real_data = False
    else:
        # Try to load Reddit hourly counts
        r_rows = _read_jsonl(reddit_log)
        r_counts = _bucket_hourly_counts(r_rows, ("ts_ingested_utc", "created_utc", "ts", "timestamp"))
        r_vec = _align_series(grid, r_counts)

        # Twitter hourly counts
        t_rows = _read_jsonl(twitter_log)
        t_counts = _bucket_hourly_counts(t_rows, ("ts_ingested_utc", "created_utc", "ts", "timestamp"))
        t_vec = _align_series(grid, t_counts)

        # Market hourly returns
        m_rows = _read_jsonl(market_log)
        m_returns = _build_market_returns(m_rows, grid)
        m_vec = _align_series(grid, m_returns)

        # Check if any is essentially empty
        if (np.nansum(np.abs(r_vec)) == 0) or (np.nansum(np.abs(t_vec)) == 0) or (np.all(np.isnan(m_vec)) or np.nansum(np.abs(np.nan_to_num(m_vec))) == 0):
            have_real_data = False
        else:
            series["reddit"] = r_vec
            series["twitter"] = t_vec
            # Replace NaNs with 0 for correlation; returns can be sparse at first hour(s)
            series["market"] = np.nan_to_num(m_vec, nan=0.0)

    if not have_real_data:
        # Deterministic demo seed
        rng = np.random.default_rng(42)
        n = lookback_h
        base = rng.normal(0, 1, size=n).cumsum()
        reddit = base + rng.normal(0, 0.3, size=n)
        twitter = np.roll(base, -1) + rng.normal(0, 0.3, size=n)  # reddit leads twitter by +1h
        market = np.roll(base, -2) * 0.6 + rng.normal(0, 0.4, size=n)  # reddit leads market by +2h (we’ll analyze pairwise)
        series = {
            "reddit": reddit,
            "twitter": twitter,
            "market": market,
        }

    # -------------------------
    # Compute pairwise stats
    # -------------------------
    pairs = [("reddit", "twitter"), ("reddit", "market"), ("twitter", "market")]
    results: List[Dict[str, Any]] = []

    # For heatmap
    mat_index = {name: i for i, name in enumerate(labels)}
    heat = np.zeros((len(labels), len(labels)), dtype=float)

    for a, b in pairs:
        x = series[a]
        y = series[b]

        # Pearson (same-time)
        r_now = _pearson(x, y)

        # Lead/lag via cross-correlation
        lag, rbest, lags_arr, r_by_lag = _cross_corr_lag(x, y, max_shift_h)

        # Permutation-test significance at the selected lag
        p_val = _perm_p_value(x, y, lag, rbest, n_perm=n_perm, rng_seed=123)
        significant = (p_val < 0.05)

        # Human text for lag
        if lag > 0:
            lag_txt = f"{a} leads by +{lag}h"
        elif lag < 0:
            lag_txt = f"{b} leads by +{abs(lag)}h"
        else:
            lag_txt = "synchronous"

        results.append({
            "pair": f"{a}–{b}",
            "lag_hours": lag,
            "r": rbest,
            "p_value": p_val,
            "significant": bool(significant),
            "pearson_same_time": r_now,
        })

        # Fill heatmap (use best |r| with sign)
        i = mat_index[a]
        j = mat_index[b]
        heat[i, j] = rbest
        heat[j, i] = rbest

        # Plot CCF curve for this pair
        try:
            fig = plt.figure(figsize=(6, 4))
            plt.plot(lags_arr, r_by_lag)
            plt.axhline(0.0, linestyle="--", linewidth=1)
            plt.axvline(lag, linestyle=":", linewidth=1)
            title = f"CCF: {a} vs {b} (best lag={lag}, r={rbest:.2f}, p={p_val:.3f})"
            plt.title(title)
            plt.xlabel("Lag (hours)  — positive: first series leads")
            plt.ylabel("Correlation")
            # Shade significant if p<0.05
            if significant:
                ylim = plt.ylim()
                plt.fill_betweenx(ylim, lag - 0.2, lag + 0.2, alpha=0.15)
            plt.tight_layout()
            out_ccf = f"artifacts/leadlag_ccf_{a}_vs_{b}.png"
            plt.savefig(out_ccf, dpi=140)
            plt.close(fig)
        except Exception:
            pass

    # Save heatmap
    try:
        fig = plt.figure(figsize=(5, 4))
        im = plt.imshow(heat, vmin=-1.0, vmax=1.0)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.title("Lead–Lag Max Correlation (sign shown at best lag)")
        plt.tight_layout()
        plt.savefig("artifacts/leadlag_heatmap.png", dpi=140)
        plt.close(fig)
    except Exception:
        pass

    # JSON artifact
    payload = {
        "window_hours": lookback_h,
        "max_shift_hours": max_shift_h,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pairs": results,
        "demo": bool(demo or (not have_real_data)),
    }
    try:
        with open("models/leadlag_analysis.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

    # -------------------------
    # Markdown
    # -------------------------
    md.append(f"\n### ⏱️ Lead–Lag Analysis ({lookback_h}h, max ±{max_shift_h}h)")

    # Sort lines in a stable logical order
    def _score(p: Dict[str, Any]) -> float:
        # rank by absolute r then significance
        return (abs(float(p.get("r", 0.0))) + (0.01 if p.get("significant") else 0.0))

    for rec in sorted(results, key=_score, reverse=True):
        pair = rec.get("pair", "a–b")
        rbest = float(rec.get("r", float("nan")))
        lag = int(rec.get("lag_hours", 0))
        pval = float(rec.get("p_value", 1.0))
        sig = "✅" if rec.get("significant") else "❌"

        if lag > 0:
            lead_txt = f"leads by +{lag}h"
        elif lag < 0:
            lead_txt = f"leads by +{abs(lag)}h (second series)"
        else:
            lead_txt = "synchronous"

        md.append(f"{pair:<18} → r={rbest:.2f} | {lead_txt} [p={pval:.2f} {sig}]")

    md.append("_Footer: Lead/lag via cross-correlation; significance via permutation test (p<0.05)._")

# For explicit imports in other modules/tests
__all__ = [
    "_cross_corr_lag",
    "append",
]