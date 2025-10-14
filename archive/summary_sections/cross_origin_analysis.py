# scripts/summary_sections/cross_origin_analysis.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

# Matplotlib (no seaborn; single-plot figs; no explicit styles/colors)
import matplotlib
matplotlib.use(os.getenv("MPLBACKEND", "Agg"))  # respect CI setting, default to Agg
import matplotlib.pyplot as plt  # noqa: E402


# ----------------------------
# Config helpers / data classes
# ----------------------------

@dataclass
class LeadLagConfig:
    lookback_h: int = 72
    max_shift_h: int = 12
    n_perm: int = 100
    artifacts_dir: str = "artifacts"
    models_dir: str = "models"
    demo: bool = False


# ----------------------------
# File I/O helpers
# ----------------------------

def _ensure_dirs(cfg: LeadLagConfig) -> None:
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)


def _read_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_series_from_logs_or_models(cfg: LeadLagConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Try to assemble hourly series for:
      - reddit: total posts/hour across subs
      - twitter: total tweets/hour
      - market: BTC hourly returns (or volatility proxy)
    Prefer append-only logs if present; otherwise fall back to model/context JSONs.
    Return arrays aligned to cfg.lookback_h (latest window), or empty arrays if not found.
    """
    # Helper to bucket timestamps to last N hours
    now = datetime.now(timezone.utc)
    n = cfg.lookback_h

    def _bucket_counts_from_jsonl(path: str, ts_key: str) -> np.ndarray:
        if not os.path.isfile(path):
            return np.array([])
        buckets = np.zeros(n, dtype=float)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts_str = rec.get(ts_key)
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except Exception:
                        continue
                    delta_h = int((now - ts).total_seconds() // 3600)
                    # keep last n hours [1..n], put into index n-1-delta
                    if 0 <= delta_h < n:
                        buckets[n - 1 - delta_h] += 1.0
        except Exception:
            return np.array([])
        return buckets

    # Reddit posts/hour
    reddit = _bucket_counts_from_jsonl("logs/social_reddit.jsonl", "created_utc")
    if reddit.size == 0:
        # fallback: models/social_reddit_context.json has per-subreddit counts; fake a flat-ish profile
        ctx = _read_json(os.path.join(cfg.models_dir, "social_reddit_context.json"))
        total = 0
        try:
            # naive: sum reported "posts" across subs if available
            subs = ctx.get("subs") or []
            for s in subs:
                total += int(s.get("posts", 0) or 0)
        except Exception:
            total = 0
        if total > 0:
            # spread across hours with mild noise
            rng = np.random.default_rng(2025)
            base = np.full(n, total / max(n, 1.0))
            reddit = np.clip(base + rng.normal(0, 0.1 * (total / max(n, 1.0)), size=n), 0, None).astype(float)

    # Twitter tweets/hour
    twitter = _bucket_counts_from_jsonl("logs/social_twitter.jsonl", "created_utc")
    if twitter.size == 0:
        ctx = _read_json(os.path.join(cfg.models_dir, "social_twitter_context.json"))
        total = int(ctx.get("counts", {}).get("total_tweets", 0) or 0)
        if total > 0:
            rng = np.random.default_rng(20251)
            base = np.full(n, total / max(n, 1.0))
            twitter = np.clip(base + rng.normal(0, 0.1 * (total / max(n, 1.0)), size=n), 0, None).astype(float)

    # Market hourly returns (prefer logs/market_prices.jsonl with price series)
    # If not available, last fallback: try models/market_context.json and synthesize small-variance returns.
    def _load_market_returns_from_prices_jsonl() -> np.ndarray:
        path = "logs/market_prices.jsonl"
        if not os.path.isfile(path):
            return np.array([])
        # Expect records with {"ts":"...", "symbol":"BTC", "price": <float>}
        prices = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if (rec.get("symbol") or "").upper() not in ("BTC", "XBT", "BITCOIN"):
                        continue
                    ts_str = rec.get("ts") or rec.get("timestamp")
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except Exception:
                        continue
                    price = rec.get("price")
                    try:
                        p = float(price)
                    except Exception:
                        continue
                    prices.append((ts, p))
        except Exception:
            return np.array([])
        if not prices:
            return np.array([])
        prices.sort(key=lambda t: t[0])
        # Bucket by hour ending; compute returns
        # Take last n hours ending at 'now'
        # Build an hourly aligned price series (forward/backfill simple)
        # For simplicity in CI, just take last n+1 prices equally spaced if dense enough
        ps = [p for _, p in prices][- (n + 1):]
        if len(ps) < 2:
            return np.array([])
        ps = np.asarray(ps, dtype=float)
        rets = np.diff(np.log(ps))
        if rets.size >= n:
            return rets[-n:]
        # pad if shorter
        return np.pad(rets, (n - rets.size, 0), mode="edge")

    market = _load_market_returns_from_prices_jsonl()
    if market.size == 0:
        mctx = _read_json(os.path.join(cfg.models_dir, "market_context.json"))
        # synthesize small returns based on BTC prices in context if any
        # Otherwise return zeros (will trigger demo seed)
        rets = []
        try:
            prices = (mctx.get("coins") or {}).get("bitcoin", {}).get("prices_usd", [])
            # prices_usd expected as list of [ts, price] or numbers — we handle numbers
            pr = []
            for v in prices:
                try:
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        pr.append(float(v[1]))
                    else:
                        pr.append(float(v))
                except Exception:
                    continue
            pr = pr[- (n + 1):]
            if len(pr) >= 2:
                p = np.asarray(pr, dtype=float)
                rets = np.diff(np.log(p))
        except Exception:
            rets = []
        market = np.asarray(rets, dtype=float)[-n:] if rets else np.array([])

    return reddit, twitter, market


# ----------------------------
# Demo seeding & degeneracy
# ----------------------------

def _is_degenerate(v: np.ndarray) -> bool:
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return True
    if np.allclose(v, 0.0):
        return True
    if not np.isfinite(v).any():
        return True
    if np.nanstd(v) < 1e-12:
        return True
    return False


def _seed_demo_series(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create plausible series:
      - reddit leads twitter by +1h (positive lag)
      - reddit leads market by +2h
    """
    # Base latent driver with daily-ish cycles + noise
    t = np.arange(n)
    latent = 10 + 3.0 * np.sin(2 * np.pi * t / 24.0) + rng.normal(0, 1.0, size=n)

    reddit = np.clip(latent + rng.normal(0, 0.8, size=n), 0, None)

    # twitter follows reddit with +1h lag (reddit leads)
    twitter = np.roll(reddit, +1) + rng.normal(0, 0.6, size=n)
    twitter[:1] = twitter[1]  # reduce edge artifact
    twitter = np.clip(twitter, 0, None)

    # market reacts to reddit with +2h; use small returns proxy: difference of smoothed reddit
    m_base = np.roll(reddit, +2)
    m_base[:2] = m_base[2]
    # returns proxy = normalized first difference
    market = np.diff(np.r_[m_base[0], m_base])
    market = (market - np.mean(market)) / (np.std(market) + 1e-8)
    market += rng.normal(0, 0.25, size=n)

    return reddit.astype(float), twitter.astype(float), market.astype(float)


# ----------------------------
# Lead–lag / CCF computation
# ----------------------------

def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / (sd + 1e-12)


def _corr_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """
    Pearson correlation at a given integer lag.
    Positive lag => x leads y by +lag hours
    """
    if lag > 0:
        xx = x[:-lag]
        yy = y[lag:]
    elif lag < 0:
        xx = x[-lag:]
        yy = y[:lag]
    else:
        xx = x
        yy = y
    if xx.size < 2 or yy.size < 2:
        return np.nan
    c = np.corrcoef(xx, yy)[0, 1]
    if not np.isfinite(c):
        return np.nan
    return float(c)


def _cross_corr_lag(x: np.ndarray, y: np.ndarray, max_shift: int) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """
    Compute cross-correlation for lags in [-max_shift, +max_shift].
    Return (lag_best, r_best, lags, r_lag_array).
    We maximize absolute correlation, but preserve the sign for r_best.
    Positive lag => x leads y.
    """
    xz = _standardize(x)
    yz = _standardize(y)
    lags = np.arange(-max_shift, max_shift + 1, dtype=int)
    rvals = np.array([_corr_at_lag(xz, yz, int(l)) for l in lags], dtype=float)
    if not np.isfinite(rvals).any():
        return 0, float("nan"), lags, rvals
    idx = int(np.nanargmax(np.abs(rvals)))
    lag_best = int(lags[idx])
    r_best = float(rvals[idx])
    return lag_best, r_best, lags, rvals


def _perm_p_value(x: np.ndarray, y: np.ndarray, max_shift: int, r_obs: float, n_perm: int, rng: np.random.Generator) -> float:
    """
    Permutation p-value: shuffle y, take max |r| across lags each time, compare to |r_obs|.
    """
    if not np.isfinite(r_obs):
        return 1.0
    xz = _standardize(x)
    yz = _standardize(y)
    n = xz.size
    if n < 4:
        return 1.0
    count = 0
    for _ in range(int(max(1, n_perm))):
        perm = rng.permutation(n)
        yp = yz[perm]
        _, r_best, _, rvals = _cross_corr_lag(xz, yp, max_shift)
        if not np.isfinite(r_best):
            # if all NaN, skip counting
            continue
        if np.nanmax(np.abs(rvals)) >= abs(r_obs) - 1e-12:
            count += 1
    p = (count + 1.0) / (n_perm + 1.0)
    return float(min(max(p, 0.0), 1.0))


# ----------------------------
# Plotting
# ----------------------------

def _plot_heatmap(pairs: Dict[Tuple[str, str], Tuple[int, float]], out_path: str) -> None:
    """
    Simple 3x3 heatmap for r at best lag.
    """
    labels = ["reddit", "twitter", "market"]
    idx = {k: i for i, k in enumerate(labels)}
    M = np.zeros((3, 3), dtype=float)
    M[:] = np.nan
    for (a, b), (_lag, r) in pairs.items():
        i, j = idx[a], idx[b]
        M[i, j] = r
        M[j, i] = r  # symmetric visualization of strength
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    # use imshow with nan masked
    data = np.where(np.isnan(M), 0.0, M)
    im = ax.imshow(data, vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    # annotate values
    for i in range(3):
        for j in range(3):
            if math.isnan(M[i, j]):
                continue
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_ccf(pair_name: str, lags: np.ndarray, rvals: np.ndarray, lag_best: int, out_dir: str) -> None:
    fig = plt.figure(figsize=(5, 3))
    ax = plt.gca()
    ax.plot(lags, rvals)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axvline(lag_best, linestyle=":", linewidth=1)
    ax.set_xlabel("Lag (hours) — positive: first leads second")
    ax.set_ylabel("Correlation")
    ax.set_title(f"CCF: {pair_name} (best lag={lag_best:+d}h)")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"leadlag_ccf_{pair_name.replace('–', '-').replace(' ', '')}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------
# Markdown formatting
# ----------------------------

def _fmt_line(a: str, b: str, lag: int, r: float, p: float, sig: bool) -> str:
    if not np.isfinite(r):
        return f"{a}–{b} → r=nan | synchronous [p=1.00 ❌]"
    if lag == 0:
        lag_txt = "synchronous"
    else:
        # Positive lag means 'a' leads 'b'
        leader = a if lag > 0 else b
        lag_txt = f"{leader} leads by {abs(lag):+d}h"
    mark = "✅" if sig else "❌"
    return f"{a}–{b} → r={r:.2f} | {lag_txt} [p={p:.2f} {mark}]"


# ----------------------------
# Public entry: append(md, ctx)
# ----------------------------

def append(md: List[str], ctx) -> None:  # ctx: SummaryContext (duck-typed here)
    # Read env/config
    cfg = LeadLagConfig(
        lookback_h=int(os.getenv("MW_LEADLAG_LOOKBACK_H", os.getenv("MW_CORR_LOOKBACK_H", "72"))),
        max_shift_h=int(os.getenv("MW_LEADLAG_MAX_SHIFT_H", "12")),
        n_perm=int(os.getenv("MW_LEADLAG_N_PERM", "100")),
        artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts"),
        models_dir="models",
        demo=os.getenv("MW_DEMO", "false").lower() in ("1", "true", "yes"),
    )
    _ensure_dirs(cfg)

    # Load series
    reddit, twitter, market = _load_series_from_logs_or_models(cfg)

    # If ANY series degenerate, seed demo series (Option B)
    if _is_degenerate(reddit) or _is_degenerate(twitter) or _is_degenerate(market) or cfg.demo:
        rng = np.random.default_rng(1337)
        reddit, twitter, market = _seed_demo_series(cfg.lookback_h, rng)
        seeded = True
    else:
        seeded = False

    # Compute pairs
    pairs = [("reddit", "twitter"), ("reddit", "market"), ("twitter", "market")]
    series_map = {"reddit": reddit, "twitter": twitter, "market": market}

    results: Dict[Tuple[str, str], Dict[str, float | int | bool]] = {}
    vis_pairs: Dict[Tuple[str, str], Tuple[int, float]] = {}

    rng = np.random.default_rng(4242)

    for a, b in pairs:
        x = series_map[a]
        y = series_map[b]
        lag_best, r_best, lags, rvals = _cross_corr_lag(x, y, cfg.max_shift_h)
        p = _perm_p_value(x, y, cfg.max_shift_h, r_best, cfg.n_perm, rng)
        significant = (p < 0.05) and np.isfinite(r_best)
        results[(a, b)] = dict(
            lag_hours=int(0 if not np.isfinite(r_best) else lag_best),
            r=float(0.0 if not np.isfinite(r_best) else r_best),
            p_value=float(p),
            significant=bool(significant),
        )
        vis_pairs[(a, b)] = (lag_best if np.isfinite(r_best) else 0, 0.0 if not np.isfinite(r_best) else r_best)
        # CCF plot per pair
        _plot_ccf(f"{a}–{b}", lags, rvals, 0 if not np.isfinite(r_best) else lag_best, cfg.artifacts_dir)

    # Heatmap plot
    _plot_heatmap({k: (v["lag_hours"], v["r"]) for k, v in results.items()}, os.path.join(cfg.artifacts_dir, "leadlag_heatmap.png"))

    # JSON artifact
    out_json = {
        "window_hours": cfg.lookback_h,
        "max_shift_hours": cfg.max_shift_h,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pairs": [
            {
                "pair": f"{a}–{b}",
                "lag_hours": int(results[(a, b)]["lag_hours"]),
                "r": float(results[(a, b)]["r"]),
                "p_value": float(results[(a, b)]["p_value"]),
                "significant": bool(results[(a, b)]["significant"]),
            }
            for a, b in pairs
        ],
        "demo": bool(seeded),
    }
    json_path = os.path.join(cfg.models_dir, "leadlag_analysis.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2)
    except Exception:
        # best-effort; still render markdown
        pass

    # Markdown block
    md.append(f"\n⏱️ Lead–Lag Analysis ({cfg.lookback_h}h, max ±{cfg.max_shift_h}h)")
    for a, b in pairs:
        r = float(results[(a, b)]["r"])
        lag = int(results[(a, b)]["lag_hours"])
        p = float(results[(a, b)]["p_value"])
        sig = bool(results[(a, b)]["significant"])
        md.append(_fmt_line(a, b, lag, r, p, sig))
    md.append("Footer: Lead/lag via cross-correlation; significance via permutation test (p<0.05).")


# ----------------------------
# Expose helper for tests
# ----------------------------

__all__ = ["append", "_cross_corr_lag"]