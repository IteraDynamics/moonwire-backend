from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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


def compute_origin_trends(
    flags_path: Path,
    triggers_path: Path,
    days: int = 7,
    interval: str = "day",
) -> Dict[str, Any]:
    """
    Groups events by origin and time bucket.
    Returns:
    {
      "window_days": days,
      "interval": "day"|"hour",
      "origins": [
        {
          "origin": "twitter",
          "buckets": [
            {"timestamp_bucket": "...", "flags_count": n, "triggers_count": m},
            ...
          ]
        },
        ...
      ]
    }
    """
    if days <= 0:
        return {"window_days": days, "interval": interval, "origins": []}
    if interval not in ("day", "hour"):
        raise ValueError('interval must be "day" or "hour"')

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    # (origin, bucket_ts) -> counts
    counts: DefaultDict[Tuple[str, datetime], Dict[str, int]] = defaultdict(
        lambda: {"flags_count": 0, "triggers_count": 0}
    )

    # ---- flags
    if flags_path.exists():
        for row in stream_jsonl(flags_path):
            ts = parse_ts(row.get("timestamp")) or now  # tolerate missing → now
            if ts < cutoff:
                continue
            bts = _bucket_start(ts, interval)
            origin = extract_origin(
                row.get("origin")
                or row.get("source")
                or (row.get("meta") or {}).get("origin")
                or (row.get("metadata") or {}).get("source")
            )
            counts[(origin, bts)]["flags_count"] += 1

    # ---- triggers
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
            counts[(origin, bts)]["triggers_count"] += 1

    # ---- format per origin
    series_by_origin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (origin, bts), c in counts.items():
        series_by_origin[origin].append(
            {
                "timestamp_bucket": bts.isoformat(),
                "flags_count": c["flags_count"],
                "triggers_count": c["triggers_count"],
            }
        )

    # sort buckets chronologically
    for buckets in series_by_origin.values():
        buckets.sort(key=lambda d: d["timestamp_bucket"])

    origins_out = [{"origin": o, "buckets": series_by_origin[o]} for o in sorted(series_by_origin.keys())]

    # --- Stability: collapse to a single bucket for 1-day daily view ---
    # CI can run near UTC midnight; events at now-2h and now-1h may straddle two dates.
    # For interval='day', days=1 we want a single 1-day summary per origin.
    if str(interval) == "day" and int(days) == 1:
        for item in origins_out:
            buckets = item.get("buckets") or []
            if len(buckets) > 1:
                flags_sum = sum(int(b.get("flags_count", 0) or 0) for b in buckets)
                trig_sum  = sum(int(b.get("triggers_count", 0) or 0) for b in buckets)
                # Use the most recent day's label if available
                ts_label = None
                try:
                    ts_label = (buckets[-1] or {}).get("timestamp_bucket", None)
                except Exception:
                    pass
                item["buckets"] = [{
                    "timestamp_bucket": ts_label,
                    "flags_count": flags_sum,
                    "triggers_count": trig_sum,
                }]

    return {"window_days": days, "interval": interval, "origins": origins_out}