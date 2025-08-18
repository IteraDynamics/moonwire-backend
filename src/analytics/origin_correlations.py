from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Dict, Any, List, Tuple, DefaultDict

from src.analytics.origin_utils import stream_jsonl, extract_origin, parse_ts


def _bucket_start(dt: datetime, interval: str) -> datetime:
    if interval == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    else:
        raise ValueError('interval must be "day" or "hour"')


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx == 0.0 or syy == 0.0:
        # constant series -> undefined correlation; treat as 0
        return 0.0
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return sxy / (sqrt(sxx) * sqrt(syy))


def compute_origin_correlations(
    flags_path: Path,
    triggers_path: Path,   # kept for symmetry/future use; correlations use flags activity
    days: int = 7,
    interval: str = "day",
) -> Dict[str, Any]:
    """
    Bucket flag events by time (day|hour), build aligned time series per origin
    over *observed buckets only*, then compute pairwise Pearson correlations.

    Returns:
    {
      "window_days": 7,
      "interval": "day",
      "origins": ["reddit","rss_news","twitter"],
      "pairs": [{"a":"reddit","b":"twitter","correlation":0.82}, ...]
    }
    """
    if days <= 0:
        return {"window_days": days, "interval": interval, "origins": [], "pairs": []}
    if interval not in ("day", "hour"):
        raise ValueError('interval must be "day" or "hour"')

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    # Collect counts and the set of actually observed buckets in-window
    counts: DefaultDict[Tuple[str, datetime], int] = defaultdict(int)
    origins_set: set[str] = set()
    observed_buckets: set[datetime] = set()

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
            counts[(origin, bts)] += 1
            origins_set.add(origin)
            observed_buckets.add(bts)

    # No data → no pairs
    if not origins_set or not observed_buckets:
        return {"window_days": days, "interval": interval, "origins": [], "pairs": []}

    buckets_sorted = sorted(observed_buckets)
    origins_sorted = sorted(origins_set)

    # Build aligned vectors per origin across observed buckets only; skip all-zero series
    vectors: Dict[str, List[float]] = {}
    for o in origins_sorted:
        vec = [float(counts.get((o, b), 0)) for b in buckets_sorted]
        if any(v != 0.0 for v in vec):
            vectors[o] = vec

    origins = sorted(vectors.keys())
    pairs: List[Dict[str, Any]] = []
    for a, b in combinations(origins, 2):
        r = _pearson(vectors[a], vectors[b])
        pairs.append({"a": a, "b": b, "correlation": round(r, 3)})

    # Sort strongest first
    pairs.sort(key=lambda p: p["correlation"], reverse=True)

    return {"window_days": days, "interval": interval, "origins": origins, "pairs": pairs}
