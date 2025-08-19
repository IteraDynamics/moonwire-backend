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
    """Forward-ordered list of bucket starts covering the full window."""
    if interval == "day":
        start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        step = timedelta(days=1)
        count = days
    else:  # hour
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


def _mean_std(values: List[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n  # population std
    return m, sqrt(var)


def compute_bursts(
    flags_path: Path,
    triggers_path: Path,
    days: int = 7,
    interval: str = "hour",
    z_thresh: float = 2.0,
) -> Dict[str, Any]:
    """
    Detect bursty buckets per origin using z-score against the window baseline.

    Returns:
    {
      "window_days": 7,
      "interval": "hour",
      "origins": [
        {
          "origin": "twitter",
          "bursts": [
            {"timestamp_bucket": "...Z", "count": 42, "z_score": 3.1},
            ...
          ]
        },
        ...
      ]
    }
    """
    if days <= 0:
        return {"window_days": days, "interval": interval, "origins": []}
    if interval not in ("hour", "day"):
        raise ValueError('interval must be "hour" or "day"')
    if z_thresh < 0:
        z_thresh = 0.0

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    buckets = _bucket_range(now, days, interval)

    # (origin -> bucket -> count), summing flags + triggers
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

    # triggers (optional)
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

    # Build burst lists
    origins_out: List[Dict[str, Any]] = []
    for origin in sorted(per_origin.keys()):
        series = [float(per_origin[origin].get(b, 0)) for b in buckets]  # include zeros → real baseline
        mean, std = _mean_std(series)
        if std == 0.0:
            bursts = []
        else:
            bursts = []
            for b in buckets:
                c = per_origin[origin].get(b, 0)
                z = (c - mean) / std if std > 0 else 0.0
                if z >= z_thresh:
                    bursts.append({
                        "timestamp_bucket": b.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
                        "count": int(c),
                        "z_score": round(float(z), 3)
                    })

        origins_out.append({"origin": origin, "bursts": bursts})

    return {"window_days": days, "interval": interval, "origins": origins_out}
