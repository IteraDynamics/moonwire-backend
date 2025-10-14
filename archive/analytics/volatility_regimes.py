from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import sqrt
from pathlib import Path
from typing import Dict, Any, List, DefaultDict

from src.analytics.origin_utils import stream_jsonl, extract_origin, parse_ts


def _bucket_start(dt: datetime, interval: str) -> datetime:
    if interval == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    else:
        raise ValueError('interval must be "hour" or "day"')


def _bucket_range(now: datetime, days: int, interval: str) -> List[datetime]:
    """Forward-ordered, full-window bucket starts (includes zeros -> true baseline)."""
    if interval == "day":
        start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        step = timedelta(days=1)
        count = days
    else:
        hours = days * 24
        start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours - 1)
        step = timedelta(hours=1)
        count = hours

    out = []
    cur = start
    for _ in range(count):
        out.append(cur)
        cur = cur + step
    return out


def _mean_std(vals: List[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n  # population std
    return m, sqrt(var)


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Nearest-rank style percentile in [0,1]; safe for small n."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    # index in [0, n-1]
    idx = int(round(q * (n - 1)))
    return sorted_vals[idx]


def compute_volatility_regimes(
    flags_path: Path,
    triggers_path: Path,
    days: int = 30,
    interval: str = "hour",
    lookback: int = 72,        # number of buckets for rolling stats
    q_calm: float = 0.33,
    q_turb: float = 0.80
) -> Dict[str, Any]:
    """
    Build per-origin counts (flags+triggers), compute current rolling std over
    the last `lookback` buckets as vol_metric, then label each origin via
    cross-origin quantiles: <= q_calm => calm, >= q_turb => turbulent, else normal.
    """
    if days <= 0:
        return {"window_days": days, "interval": interval, "origins": []}
    if interval not in ("hour", "day"):
        raise ValueError('interval must be "hour" or "day"')
    if lookback <= 1:
        raise ValueError("lookback must be >= 2")
    if not (0 <= q_calm <= 1 and 0 <= q_turb <= 1):
        raise ValueError("quantiles must be in [0,1]")

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    buckets = _bucket_range(now, days, interval)

    # (origin -> bucket -> count), combining flags + triggers
    per_origin: DefaultDict[str, Dict[datetime, int]] = defaultdict(lambda: defaultdict(int))

    # flags
    if flags_path.exists():
        for row in stream_jsonl(flags_path):
            ts = parse_ts(row.get("timestamp")) or now
            if ts < cutoff:
                continue
            bts = _bucket_start(ts, interval)
            origin = extract_origin(
                row.get("origin")
                or row.get("source")
                or (row.get("meta") or {}).get("origin")
                or (row.get("metadata") or {}).get("source")
            )
            per_origin[origin][bts] += 1

    # triggers
    if triggers_path.exists():
        for row in stream_jsonl(triggers_path):
            ts = parse_ts(row.get("timestamp")) or now
            if ts < cutoff:
                continue
            bts = _bucket_start(ts, interval)
            origin = extract_origin(
                row.get("origin")
                or row.get("source")
                or (row.get("meta") or {}).get("origin")
                or (row.get("metadata") or {}).get("source")
            )
            per_origin[origin][bts] += 1

    # Compute current rolling stats per origin
    results: List[Dict[str, Any]] = []
    vol_values: List[float] = []

    for origin in sorted(per_origin.keys()):
        series = [float(per_origin[origin].get(b, 0)) for b in buckets]
        window = series[-lookback:] if len(series) >= lookback else series
        mean, std = _mean_std(window)
        vol_metric = std  # rolling std choice for simplicity & determinism
        vol_values.append(vol_metric)
        results.append({
            "origin": origin,
            "vol_metric": round(vol_metric, 6),
            "regime": "normal",  # to be assigned
            "stats": {"mean": round(mean, 6), "std": round(std, 6)},
        })

    # Label regimes via cross-origin quantiles of current vol_metric
    vol_sorted = sorted(vol_values)
    calm_cut = _percentile(vol_sorted, q_calm)
    turb_cut = _percentile(vol_sorted, q_turb)

    for row in results:
        v = row["vol_metric"]
        if v <= calm_cut:
            row["regime"] = "calm"
        elif v >= turb_cut:
            row["regime"] = "turbulent"
        else:
            row["regime"] = "normal"

    return {"window_days": days, "interval": interval, "origins": results}
