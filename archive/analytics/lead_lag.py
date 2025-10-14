from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import permutations
from math import sqrt
from pathlib import Path
from typing import Dict, Any, List, Tuple, DefaultDict, Iterable

from src.analytics.origin_utils import stream_jsonl, extract_origin, parse_ts


def _bucket_start(dt: datetime, interval: str) -> datetime:
    if interval == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    raise ValueError('interval must be "hour" or "day"')


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx == 0.0 or syy == 0.0:
        return 0.0
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return sxy / (sqrt(sxx) * sqrt(syy))


def _series_from_jsonl(path: Path, days: int, interval: str, use_field: str) -> Tuple[Dict[str, Dict[datetime, int]], Dict[str, int]]:
    """
    Returns:
      per_origin: {origin: {bucket_ts: count}}
      totals:     {origin: total_count}
    Reads only the requested stream: flags or triggers.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    per_origin: DefaultDict[str, Dict[datetime, int]] = defaultdict(lambda: defaultdict(int))
    totals: DefaultDict[str, int] = defaultdict(int)

    if not path.exists():
        return {}, {}

    for row in stream_jsonl(path):
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
        totals[origin] += 1

    return dict(per_origin), dict(totals)


def _aligned_vectors(a_map: Dict[datetime, int], b_map: Dict[datetime, int], lag: int, step: timedelta) -> Tuple[List[float], List[float]]:
    """
    For lag L: compare A[t] with B[t+L*step]. We only keep timestamps where both exist.
    """
    if lag == 0:
        common = sorted(set(a_map.keys()) & set(b_map.keys()))
        return [float(a_map[t]) for t in common], [float(b_map[t]) for t in common]

    out_x: List[float] = []
    out_y: List[float] = []
    for t in a_map.keys():
        t2 = t + lag * step
        if t2 in b_map:
            out_x.append(float(a_map[t]))
            out_y.append(float(b_map[t2]))
    return out_x, out_y


def compute_lead_lag(
    flags_path: Path,
    triggers_path: Path,
    days: int = 7,
    interval: str = "hour",      # {"hour","day"}
    max_lag: int = 24,           # number of intervals
    use: str = "flags"           # {"flags","triggers"}
) -> Dict[str, Any]:
    """
    Returns lead/lag across ordered origin pairs using cross-correlation.
    positive best_lag => A leads B by best_lag intervals
    negative best_lag => B leads A by |best_lag| intervals
    """
    if interval not in ("hour", "day"):
        raise ValueError('interval must be "hour" or "day"')
    if days <= 0:
        return {"window_days": days, "interval": interval, "max_lag": max_lag, "origins": [], "pairs": []}
    if max_lag < 0:
        max_lag = 0
    if use not in ("flags", "triggers"):
        raise ValueError('use must be "flags" or "triggers"')

    # Select which stream to analyze
    if use == "flags":
        per_origin, totals = _series_from_jsonl(flags_path, days, interval, "flags")
    else:
        per_origin, totals = _series_from_jsonl(triggers_path, days, interval, "triggers")

    if not per_origin:
        return {"window_days": days, "interval": interval, "max_lag": max_lag, "origins": [], "pairs": []}

    # Limit to top 20 origins by total activity, then name
    origins_sorted = sorted(totals.keys(), key=lambda o: (-totals[o], o))[:20]

    step = timedelta(hours=1) if interval == "hour" else timedelta(days=1)

    pairs: List[Dict[str, Any]] = []
    for a, b in permutations(origins_sorted, 2):  # ordered pairs
        a_map = per_origin.get(a, {})
        b_map = per_origin.get(b, {})
        if not a_map or not b_map:
            continue

        best_r = 0.0
        best_L = None

        for L in range(-max_lag, max_lag + 1):
            x, y = _aligned_vectors(a_map, b_map, L, step)
            if len(x) < 3:
                continue
            r = _pearson(x, y)
            if abs(r) > abs(best_r):
                best_r = r
                best_L = L

        if best_L is None or best_r == 0.0:
            continue

        leader = a if best_L > 0 else (b if best_L < 0 else "tie")
        pairs.append({
            "a": a,
            "b": b,
            "best_lag": int(best_L),
            "correlation": round(float(best_r), 3),
            "leader": leader
        })

    # Sort by |correlation| desc, then by names
    pairs.sort(key=lambda p: (-abs(p["correlation"]), p["a"], p["b"]))
    return {
        "window_days": days,
        "interval": interval,
        "max_lag": max_lag,
        "use": use,
        "origins": origins_sorted,
        "pairs": pairs
    }
