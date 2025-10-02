# scripts/summary_sections/cross_origin_correlation.py
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # headless in CI
import matplotlib.pyplot as plt  # noqa: E402

# --- Common helpers (best-effort import; fallbacks keep CI resilient) ---
try:
    from .common import (
        SummaryContext,
        ensure_dir,
        _iso,
        _load_jsonl,
        _write_json,
    )
except Exception:
    # Minimal fallbacks so the section still runs in isolation.
    @dataclass
    class SummaryContext:  # type: ignore
        logs_dir: Optional[os.PathLike] = None
        models_dir: Optional[os.PathLike] = None
        artifacts_dir: Optional[os.PathLike] = None
        is_demo: bool = False
        caches: Dict = None
        candidates: List[str] = None
        origins_rows: List[Dict] = None
        yield_data: Dict = None

    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def _iso(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _load_jsonl(path) -> List[Dict]:
        if not os.path.exists(path):
            return []
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def _write_json(path, obj) -> None:
        ensure_dir(os.path.dirname(str(path)))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)


# ------------------------------ Utilities ------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _floor_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)


def _parse_ts(x) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        # Accept seconds or ms
        try:
            if x > 1e12:  # ms
                x = x / 1000.0
            return datetime.fromtimestamp(float(x), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(x, str):
        try:
            # Basic ISO-8601 handling
            if x.endswith("Z"):
                x = x[:-1] + "+00:00"
            return datetime.fromisoformat(x).astimezone(timezone.utc)
        except Exception:
            return None
    if isinstance(x, datetime):
        return x.astimezone(timezone.utc)
    return None


def _pearson(x: List[float], y: List[float]) -> Optional[float]:
    n = min(len(x), len(y))
    if n == 0:
        return None
    xa, ya = x[:n], y[:n]
    mx = sum(xa) / n
    my = sum(ya) / n
    vx = sum((v - mx) ** 2 for v in xa)
    vy = sum((v - my) ** 2 for v in ya)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((xa[i] - mx) * (ya[i] - my) for i in range(n))
    r = cov / math.sqrt(vx * vy)
    # Clamp numerical noise
    return max(-1.0, min(1.0, r))


def _cross_corr_best_lag(a: List[float], b: List[float], max_lag: int = 6) -> Tuple[int, Optional[float]]:
    """
    Returns (best_lag_in_hours, corr_at_best_lag).

    Convention: positive lag k means **A leads** by k hours (A shifted forward aligns with B).
    """
    best_lag = 0
    best_val = None
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            x, y = a, b
        elif lag > 0:
            # A leads: compare a[:-lag] with b[lag:]
            if lag >= len(a) or lag >= len(b):
                continue
            x, y = a[:-lag], b[lag:]
        else:
            # B leads by -lag: compare a[-lag:] with b[:lag*-1]
            k = -lag
            if k >= len(a) or k >= len(b):
                continue
            x, y = a[k:], b[:-k]
        r = _pearson(x, y)
        if r is None:
            continue
        if (best_val is None) or (abs(r) > abs(best_val)):
            best_val = r
            best_lag = lag
    return best_lag, best_val


def _series_from_counts(counts_by_hour: Dict[datetime, int], buckets: List[datetime]) -> List[float]:
    return [float(counts_by_hour.get(b, 0)) for b in buckets]


def _btc_returns_from_market_jsonl(path_jsonl: str, buckets: List[datetime]) -> Optional[List[float]]:
    """
    Read logs/market_prices.jsonl with rows like:
      {"t": 1696114800, "price": 60001.2} OR
      {"ts":"...","price":...}
    Produce % returns per hour aligned to buckets (len == len(buckets))
    """
    if not os.path.exists(path_jsonl):
        return None
    rows = _load_jsonl(path_jsonl)
    pts: Dict[datetime, float] = {}
    for r in rows:
        ts = _parse_ts(r.get("ts") or r.get("t"))
        if ts is None:
            continue
        price = r.get("price")
        try:
            price = float(price)
        except Exception:
            continue
        pts[_floor_hour(ts)] = price
    if not pts:
        return None

    # Build return series aligned to buckets
    out: List[float] = []
    prev_price = None
    for b in buckets:
        p = pts.get(b)
        if p is None:
            out.append(0.0)
        else:
            if prev_price is None or prev_price == 0:
                out.append(0.0)
            else:
                out.append((p - prev_price) / prev_price)
            prev_price = p
    return out


# ------------------------------ Core section ----------------------------------


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Compute cross-origin correlations between Reddit (posts), Twitter (tweets), and Market (BTC 1h returns).
    Emits:
      - models/cross_origin_correlation.json
      - artifacts/corr_heatmap.png
      - artifacts/corr_leadlag.png
    Adds a markdown block to the CI summary.
    """
    lookback_h = _env_int("MW_CORR_LOOKBACK_H", 72)
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=lookback_h)

    models_dir = str(getattr(ctx, "models_dir", "models") or "models")
    logs_dir = str(getattr(ctx, "logs_dir", "logs") or "logs")
    arts_dir = str(getattr(ctx, "artifacts_dir", "artifacts") or "artifacts")

    ensure_dir(models_dir)
    ensure_dir(logs_dir)
    ensure_dir(arts_dir)

    # Build hourly bucket timeline
    buckets: List[datetime] = []
    cur = _floor_hour(start)
    end = _floor_hour(now)
    while cur <= end:
        buckets.append(cur)
        cur += timedelta(hours=1)

    # ---- Reddit counts ----
    reddit_counts: Dict[datetime, int] = Counter()
    reddit_log = os.path.join(logs_dir, "social_reddit.jsonl")
    if os.path.exists(reddit_log):
        for r in _load_jsonl(reddit_log):
            ts = _parse_ts(r.get("created_utc") or r.get("ts") or r.get("timestamp"))
            if ts is None:
                continue
            if ts < start or ts > now:
                continue
            reddit_counts[_floor_hour(ts)] += 1

    # ---- Twitter counts ----
    twitter_counts: Dict[datetime, int] = Counter()
    twitter_log = os.path.join(logs_dir, "social_twitter.jsonl")
    if os.path.exists(twitter_log):
        for r in _load_jsonl(twitter_log):
            ts = _parse_ts(r.get("created_utc") or r.get("ts") or r.get("timestamp"))
            if ts is None:
                continue
            if ts < start or ts > now:
                continue
            twitter_counts[_floor_hour(ts)] += 1

    # ---- Market returns ----
    market_jsonl = os.path.join(logs_dir, "market_prices.jsonl")
    market_ret = _btc_returns_from_market_jsonl(market_jsonl, buckets)

    reddit_series = _series_from_counts(reddit_counts, buckets)
    twitter_series = _series_from_counts(twitter_counts, buckets)

    # If market JSONL absent, fall back to zeros (still renders a section),
    # and in demo mode we’ll seed correlations.
    if market_ret is None:
        market_ret = [0.0] * len(buckets)

    have_real = any(v > 0 for v in reddit_series) or any(v > 0 for v in twitter_series)

    demo = bool(getattr(ctx, "is_demo", False)) or os.getenv("MW_DEMO", "false").lower() == "true"
    seeded = False

    if (not have_real) and demo:
        # Seed plausible demo data (deterministic pattern)
        seeded = True
        for i, b in enumerate(buckets):
            base = 10 + (i % 5)  # gentle wave
            reddit_series[i] = float(base + (1 if (i % 7 == 0) else 0))
            twitter_series[i] = float(base * 1.1 + (1 if (i % 9 == 0) else 0))
            market_ret[i] = 0.001 * math.sin(i / 6.0)
    # Compute Pearson pairwise
    pairs = {
        "reddit_twitter": _pearson(reddit_series, twitter_series),
        "reddit_market": _pearson(reddit_series, market_ret),
        "twitter_market": _pearson(twitter_series, market_ret),
    }

    # Lead-lag (±6h)
    max_lag = 6
    lag_rt, _ = _cross_corr_best_lag(reddit_series, twitter_series, max_lag)
    lag_rm, _ = _cross_corr_best_lag(reddit_series, market_ret, max_lag)
    lag_tm, _ = _cross_corr_best_lag(twitter_series, market_ret, max_lag)

    def fmt_lag(lag: int) -> str:
        if lag == 0:
            return "0h"
        sign = "+" if lag > 0 else "-"
        return f"{sign}{abs(lag)}h"

    lead_lag = {
        "reddit→twitter": fmt_lag(lag_rt),
        "reddit→market": fmt_lag(lag_rm),
        "twitter→market": fmt_lag(lag_tm),
    }

    # Persist JSON
    out_json = {
        "window_hours": lookback_h,
        "generated_at": _iso(now),
        "pearson": {k: (None if v is None else round(float(v), 2)) for k, v in pairs.items()},
        "lead_lag": lead_lag,
        "demo": bool(seeded),
    }
    _write_json(os.path.join(models_dir, "cross_origin_correlation.json"), out_json)

    # Plots
    try:
        # Heatmap
        labels = ["reddit", "twitter", "market"]
        mat = [
            [1.0, pairs["reddit_twitter"] or 0.0, pairs["reddit_market"] or 0.0],
            [pairs["reddit_twitter"] or 0.0, 1.0, pairs["twitter_market"] or 0.0],
            [pairs["reddit_market"] or 0.0, pairs["twitter_market"] or 0.0, 1.0],
        ]
        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()
        im = ax.imshow(mat, vmin=-1, vmax=1)
        ax.set_xticks(range(3), labels)
        ax.set_yticks(range(3), labels)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(arts_dir, "corr_heatmap.png"), dpi=120)
        plt.close(fig)
    except Exception:
        pass

    try:
        # Lead-lag bar chart (positive bar => left leads right)
        ll_pairs = [("reddit→twitter", lag_rt), ("reddit→market", lag_rm), ("twitter→market", lag_tm)]
        fig = plt.figure(figsize=(5, 2.5))
        ax = plt.gca()
        ax.bar([p[0] for p in ll_pairs], [p[1] for p in ll_pairs])
        ax.set_ylabel("Lead (hours)")
        ax.set_xlabel("Pair")
        fig.tight_layout()
        fig.savefig(os.path.join(arts_dir, "corr_leadlag.png"), dpi=120)
        plt.close(fig)
    except Exception:
        pass

    # Markdown
    md.append(f"\n### 🔗 Cross-Origin Correlations ({lookback_h}h)")
    def r2(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v:.2f}"
    # Interpret lag text for prose (positive means left leads)
    def lag_phrase(lag: int) -> str:
        if lag > 0:
            return f"left leads by ~{lag}h"
        if lag < 0:
            return f"right leads by ~{abs(lag)}h"
        return "synchronous"

    md.append(f"reddit–twitter   → r={r2(pairs['reddit_twitter'])} | {lag_phrase(lag_rt)}")
    md.append(f"reddit–market    → r={r2(pairs['reddit_market'])} | {lag_phrase(lag_rm)}")
    md.append(f"twitter–market   → r={r2(pairs['twitter_market'])} | {lag_phrase(lag_tm)}")