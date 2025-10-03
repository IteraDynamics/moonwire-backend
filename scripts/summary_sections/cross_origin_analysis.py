# scripts/summary_sections/cross_origin_analysis.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Tuple

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # CI-friendly
import matplotlib.pyplot as plt


# ----------------------------- Small utilities --------------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _env_f(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default


def _env_i(key: str, default: int) -> int:
    try:
        return int(float(os.getenv(key, str(default))))
    except Exception:
        return default


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _load_jsonl_counts(path: str, ts_key: str) -> Dict[str, int]:
    """Return counts per ISO hour (YYYY-mm-ddTHH:00:00Z)."""
    out: Dict[str, int] = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = row.get(ts_key) or row.get("created_utc") or row.get("ts") or row.get("timestamp")
                if not ts:
                    continue
                try:
                    # normalize to hour bucket
                    t = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
                    bucket = t.replace(minute=0, second=0, microsecond=0)
                    key = bucket.strftime("%Y-%m-%dT%H:00:00Z")
                    out[key] = out.get(key, 0) + 1
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _load_market_returns_jsonl(path: str) -> Dict[str, float]:
    """Return BTC hourly returns per hour ISO key."""
    out: Dict[str, float] = {}
    if not os.path.exists(path):
        return out
    last_price = None
    last_hour_key = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = row.get("ts_utc") or row.get("ts") or row.get("timestamp")
                price = row.get("price") or row.get("close") or row.get("value")
                if ts is None or price is None:
                    continue
                try:
                    t = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
                except Exception:
                    continue
                bucket = t.replace(minute=0, second=0, microsecond=0)
                key = bucket.strftime("%Y-%m-%dT%H:00:00Z")
                price = float(price)
                if last_price is not None and last_hour_key is not None and key != last_hour_key:
                    ret = (price - last_price) / last_price if last_price != 0 else 0.0
                    out[key] = float(ret)
                last_price = price
                last_hour_key = key
    except Exception:
        pass
    return out


def _align_series(keys: List[str], d: Dict[str, float | int]) -> np.ndarray:
    return np.array([float(d.get(k, 0.0)) for k in keys], dtype=float)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cross_corr_lag(a: np.ndarray, b: np.ndarray, max_shift: int) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """
    Return (best_lag, best_r, lags, r_by_lag).
    Convention:
      lag > 0  => series A leads by +lag hours (A precedes B)
      lag < 0  => series B leads by +|lag| hours
      lag == 0 => synchronous
    """
    n = len(a)
    if n < 3 or len(b) != n:
        return 0, float("nan"), np.zeros(1, dtype=int), np.zeros(1, dtype=float)

    # de-mean to be safe for correlation comparison
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)

    lags = np.arange(-max_shift, max_shift + 1, dtype=int)
    rvals = np.zeros_like(lags, dtype=float)

    # Definition: for a given lag L:
    #   if L > 0: correlate a[:-L] vs b[L:]
    #   if L < 0: correlate a[-L:] vs b[:L]
    #   if L == 0: correlate a vs b
    for i, L in enumerate(lags):
        if L > 0:
            aa = a0[:-L]
            bb = b0[L:]
        elif L < 0:
            aa = a0[-L:]
            bb = b0[:L]
        else:
            aa = a0
            bb = b0

        if len(aa) < 3 or len(bb) < 3:
            r = float("nan")
        else:
            if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
                r = float("nan")
            else:
                r = float(np.corrcoef(aa, bb)[0, 1])
        rvals[i] = r

    valid = ~np.isnan(rvals)
    if not np.any(valid):
        return 0, float("nan"), lags, rvals

    idx = int(np.nanargmax(np.abs(rvals)))
    return int(lags[idx]), float(rvals[idx]), lags, rvals


def _perm_test_ccf(a: np.ndarray, b: np.ndarray, lag: int, r_obs: float, n_perm: int = 100) -> float:
    """
    Permutation test: shuffle one series' order (break temporal structure) and recompute
    correlation at the chosen lag. Two-sided p-value on |r|.
    """
    if math.isnan(r_obs):
        return 1.0
    rng = np.random.default_rng(42)
    count_extreme = 0
    for _ in range(max(1, n_perm)):
        b_perm = rng.permutation(b)
        L = lag
        if L > 0:
            aa = a[:-L]
            bb = b_perm[L:]
        elif L < 0:
            aa = a[-L:]
            bb = b_perm[:L]
        else:
            aa = a
            bb = b_perm
        if len(aa) < 3 or len(bb) < 3:
            r = 0.0
        else:
            if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
                r = 0.0
            else:
                r = float(np.corrcoef(aa, bb)[0, 1])
        if abs(r) >= abs(r_obs):
            count_extreme += 1
    p = (count_extreme + 1) / (n_perm + 1)  # add-1 smoothing
    return float(p)


# ----------------------------- Core workflow ----------------------------------

@dataclass
class SeriesBundle:
    keys: List[str]
    reddit: np.ndarray
    twitter: np.ndarray
    market: np.ndarray
    demo: bool = False


def _build_hour_keys(lookback_h: int) -> List[str]:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    keys = [(now - timedelta(hours=h)).strftime("%Y-%m-%dT%H:00:00Z") for h in range(lookback_h - 1, -1, -1)]
    return keys


def _seed_demo_series(keys: List[str]) -> SeriesBundle:
    rng = np.random.default_rng(7)
    n = len(keys)

    # Seed Reddit/Twitter as moderately correlated counts; Market as smoothed noise with some correlation
    reddit = rng.poisson(50, size=n).astype(float)
    twitter = reddit * 0.6 + rng.poisson(30, size=n) * 0.4
    market = rng.normal(0, 1, size=n).cumsum()
    market = (market - np.mean(market)) / (np.std(market) + 1e-9)

    # inject simple lags: reddit leads twitter by 1h, reddit leads market by 2h
    twitter = np.roll(twitter, -1)  # reddit leads by +1h => twitter lags
    market = np.roll(market, -2)    # reddit leads by +2h => market lags

    # normalize
    def norm(v: np.ndarray) -> np.ndarray:
        v = np.array(v, dtype=float)
        return (v - v.mean()) / (v.std() + 1e-9)

    return SeriesBundle(
        keys=keys,
        reddit=norm(reddit),
        twitter=norm(twitter),
        market=norm(market),
        demo=True,
    )


def _gather_series(lookback_h: int) -> SeriesBundle:
    keys = _build_hour_keys(lookback_h)

    reddit_counts = _load_jsonl_counts("logs/social_reddit.jsonl", ts_key="ts_ingested_utc")
    twitter_counts = _load_jsonl_counts("logs/social_twitter.jsonl", ts_key="ts_ingested_utc")
    market_rets   = _load_market_returns_jsonl("logs/market_prices.jsonl")

    reddit = _align_series(keys, reddit_counts)
    twitter = _align_series(keys, twitter_counts)
    market = _align_series(keys, market_rets)

    # If everything is empty, produce demo series.
    if (reddit.sum() == 0) and (twitter.sum() == 0) and (np.allclose(market, 0.0)):
        return _seed_demo_series(keys)

    # Standardize (z-score) each
    def z(v: np.ndarray) -> np.ndarray:
        mu = float(np.mean(v))
        sd = float(np.std(v))
        return (v - mu) / (sd + 1e-9)

    return SeriesBundle(
        keys=keys,
        reddit=z(reddit),
        twitter=z(twitter),
        market=z(market),
        demo=False,
    )


def _pairwise(series: SeriesBundle) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    return [
        ("reddit–twitter", series.reddit, series.twitter),
        ("reddit–market",  series.reddit, series.market),
        ("twitter–market", series.twitter, series.market),
    ]


def _pearson_block(series: SeriesBundle) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, a, b in _pairwise(series):
        out[name.replace("–", "_")] = _pearson(a, b)
    return out


def _leadlag_block(series: SeriesBundle, max_shift_h: int, n_perm: int) -> List[Dict[str, Any]]:
    res: List[Dict[str, Any]] = []
    for name, a, b in _pairwise(series):
        lag, rbest, lags, rvals = _cross_corr_lag(a, b, max_shift_h)
        pval = _perm_test_ccf(a, b, lag, rbest, n_perm)
        res.append({
            "pair": name,
            "lag_hours": int(lag),
            "r": float(rbest),
            "p_value": float(pval),
            "significant": bool(pval < 0.05),
            "lags": lags.tolist(),
            "r_by_lag": rvals.tolist(),
        })
    return res


def _save_json(obj: Any, path: str) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _heatmap(matrix: np.ndarray, labels: List[str], title: str, outpath: str) -> None:
    _ensure_dir(outpath)
    fig = plt.figure(figsize=(4.2, 3.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                s = "n/a"
            else:
                s = f"{val:.2f}"
            ax.text(j, i, s, ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_ccf(pair: str, lags: Iterable[int], rvals: Iterable[float], outpath: str) -> None:
    _ensure_dir(outpath)
    lags = np.array(list(lags), dtype=int)
    rvals = np.array(list(rvals), dtype=float)
    fig = plt.figure(figsize=(5.2, 3.6))
    ax = fig.add_subplot(111)
    ax.plot(lags, rvals, marker="o")
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Lag (hours)  — positive = first series leads")
    ax.set_ylabel("Correlation r")
    ax.set_title(f"CCF: {pair}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# --------------------------------- Public API ---------------------------------

def append(md: List[str], ctx) -> None:  # ctx: SummaryContext (duck-typed here)
    look_h = _env_i("MW_LEADLAG_LOOKBACK_H", 72)
    max_shift = _env_i("MW_LEADLAG_MAX_SHIFT_H", 12)
    n_perm = _env_i("MW_LEADLAG_N_PERM", 100)
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")

    # Build or seed series
    series = _gather_series(look_h)

    # 1) Pearson matrix
    pearson_map = _pearson_block(series)

    # 2) Lead/Lag with permutation p-values
    results = _leadlag_block(series, max_shift, n_perm)

    # 3) Save JSON artifacts
    corr_json = {
        "window_hours": look_h,
        "generated_at": _now_utc_iso(),
        "pearson": pearson_map,
        "lead_lag": {r["pair"].replace("–", "_"): r["lag_hours"] for r in results},
        "demo": series.demo,
    }
    _save_json(corr_json, "models/cross_origin_correlation.json")

    leadlag_json = {
        "window_hours": look_h,
        "max_shift_hours": max_shift,
        "generated_at": _now_utc_iso(),
        "pairs": [
            {
                "pair": r["pair"],
                "lag_hours": r["lag_hours"],
                "r": r["r"],
                "p_value": r["p_value"],
                "significant": r["significant"],
            }
            for r in results
        ],
        "demo": series.demo,
    }
    _save_json(leadlag_json, "models/leadlag_analysis.json")

    # 4) Plots
    labels = ["reddit", "twitter", "market"]

    # Pearson heatmap (ordered: reddit, twitter, market)
    r_rt = pearson_map.get("reddit_twitter", float("nan"))
    r_rm = pearson_map.get("reddit_market", float("nan"))
    r_tm = pearson_map.get("twitter_market", float("nan"))
    pearson_matrix = np.array([
        [1.0, r_rt, r_rm],
        [r_rt, 1.0, r_tm],
        [r_rm, r_tm, 1.0],
    ], dtype=float)
    _heatmap(pearson_matrix, labels, "Pearson Correlations", os.path.join(artifacts_dir, "corr_heatmap.png"))

    # Lead–lag heatmap: matrix of r at the selected (max-|r|) lag for each pair
    # Build a 3x3 with symmetric fill; diagonal = 1.0
    r_map = {rec["pair"]: rec for rec in results}
    # Helper to fetch r for a pair irrespective of order used above
    def _r_for(a: str, b: str) -> float:
        key1 = f"{a}–{b}"
        key2 = f"{b}–{a}"
        if key1 in r_map:
            return float(r_map[key1]["r"])
        if key2 in r_map:
            return float(r_map[key2]["r"])
        return float("nan")

    leadlag_matrix = np.array([
        [1.0, _r_for("reddit", "twitter"), _r_for("reddit", "market")],
        [_r_for("twitter", "reddit"), 1.0, _r_for("twitter", "market")],
        [_r_for("market", "reddit"), _r_for("market", "twitter"), 1.0],
    ], dtype=float)
    _heatmap(leadlag_matrix, labels, "Lead–Lag (r at best lag)", os.path.join(artifacts_dir, "leadlag_heatmap.png"))

    # Max-lag (who leads) simple bar plot: map lag->signed hours per pair
    # (Keep old corr_leadlag.png for backward-compat)
    try:
        pairs_order = ["reddit–twitter", "reddit–market", "twitter–market"]
        lag_vals = [next((r["lag_hours"] for r in results if r["pair"] == p), 0) for p in pairs_order]
        fig = plt.figure(figsize=(5.6, 3.2))
        ax = fig.add_subplot(111)
        ax.bar(range(len(pairs_order)), lag_vals)
        ax.set_xticks(range(len(pairs_order)))
        ax.set_xticklabels(pairs_order, rotation=45, ha="right")
        ax.axhline(0.0, linestyle="--")
        ax.set_ylabel("Lag (hours)  — positive = first series leads")
        ax.set_title("Lead/Lag (max |r| lag)")
        fig.tight_layout()
        fig.savefig(os.path.join(artifacts_dir, "corr_leadlag.png"), dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # CCF per pair
    for r in results:
        _plot_ccf(
            r["pair"],
            r.get("lags", []),
            r.get("r_by_lag", []),
            os.path.join(artifacts_dir, f"leadlag_ccf_{r['pair'].replace('–','_')}.png"),
        )

    # 5) Markdown
    md.append(f"\n⏱️ Lead–Lag Analysis ({look_h}h, max ±{max_shift}h)")

    # Sort results by |r| desc for readability
    def _score(rec: Dict[str, Any]) -> float:
        v = rec.get("r")
        try:
            return abs(float(v))
        except Exception:
            return 0.0

    for rec in sorted(results, key=_score, reverse=True):
        pair = rec.get("pair", "a–b")
        rbest = float(rec.get("r", float("nan")))
        lag = int(rec.get("lag_hours", 0))
        pval = float(rec.get("p_value", 1.0))
        sig = "✅" if rec.get("significant") else "❌"
        # pair is "a–b" (en dash)
        try:
            a_name, b_name = pair.split("–", 1)
        except Exception:
            a_name, b_name = "first", "second"

        if lag > 0:
            lead_txt = f"{a_name} leads by +{lag}h"
        elif lag < 0:
            lead_txt = f"{b_name} leads by +{abs(lag)}h"
        else:
            lead_txt = "synchronous"

        md.append(f"{pair:<18} → r={rbest:.2f} | {lead_txt} [p={pval:.2f} {'✅' if pval < 0.05 else '❌'}]")

    if series.demo:
        md.append("Footer: Lead/lag via cross-correlation; significance via permutation test (demo).")
    else:
        md.append("Footer: Lead/lag via cross-correlation; significance via permutation test (p<0.05).")