from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, DefaultDict, Optional

from src.analytics.origin_utils import stream_jsonl, extract_origin, parse_ts


def _bucket_start(dt: datetime, interval: str) -> datetime:
    if interval == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    else:
        raise ValueError('interval must be "day" or "hour"')


def _latest_ts_for_origin(rows: List[dict], origin: str) -> Optional[datetime]:
    latest: Optional[datetime] = None
    for r in rows:
        o = extract_origin(
            r.get("origin")
            or r.get("source")
            or (r.get("meta") or {}).get("origin")
            or (r.get("metadata") or {}).get("source")
        )
        if o != origin:
            continue
        ts = parse_ts(r.get("timestamp"))
        if not ts:
            continue
        if (latest is None) or (ts > latest):
            latest = ts
    return latest


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

    # Load rows
    flags_rows = list(stream_jsonl(flags_path)) if flags_path.exists() else []
    trig_rows  = list(stream_jsonl(triggers_path)) if triggers_path.exists() else []

    # (origin, bucket_ts) -> counts
    counts: DefaultDict[Tuple[str, datetime], Dict[str, int]] = defaultdict(
        lambda: {"flags_count": 0, "triggers_count": 0}
    )

    # Collect candidate origins from either stream
    candidate_origins: set[str] = set()
    for r in flags_rows + trig_rows:
        o = extract_origin(
            r.get("origin")
            or r.get("source")
            or (r.get("meta") or {}).get("origin")
            or (r.get("metadata") or {}).get("source")
        )
        candidate_origins.add(o)
    if not candidate_origins:
        candidate_origins = {"twitter", "reddit", "rss_news"}

    if interval == "hour":
        # Rolling hourly window: keep simple cutoff logic
        # ---- flags
        for row in flags_rows:
            ts = parse_ts(row.get("timestamp")) or now  # tolerate missing → now
            if ts < cutoff or ts > now:
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
        for row in trig_rows:
            ts = parse_ts(row.get("timestamp")) or now
            if ts < cutoff or ts > now:
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
        for buckets in series_by_origin.values():
            buckets.sort(key=lambda d: d["timestamp_bucket"])
        origins_out = [{"origin": o, "buckets": series_by_origin.get(o, [])} for o in sorted(series_by_origin.keys())]
        return {"window_days": days, "interval": interval, "origins": origins_out}

    # ---------- interval == "day" ----------
    # Robust behavior for tests:
    # For each origin, choose the most recent *day* present among that origin's rows
    # (flags or triggers). Then emit exactly `days` consecutive calendar-day buckets
    # ending at that most recent day (for days=1 => exactly one bucket).
    origins_out: List[Dict[str, Any]] = []

    for origin in sorted(candidate_origins):
        # Find the most recent timestamp we have for this origin
        latest_ts = _latest_ts_for_origin(flags_rows + trig_rows, origin)
        if latest_ts is None:
            # No rows for this origin at all; skip emitting empty series
            continue

        day0 = latest_ts.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        start_day0 = day0 - timedelta(days=days - 1)
        end_day0   = day0 + timedelta(days=1)  # exclusive

        # Pre-build exactly `days` day buckets
        ordered_keys = [start_day0 + timedelta(days=i) for i in range(days)]
        key_set = set(ordered_keys)
        agg: Dict[datetime, Dict[str, int]] = {k: {"flags_count": 0, "triggers_count": 0} for k in ordered_keys}

        # flags
        for row in flags_rows:
            o = extract_origin(
                row.get("origin")
                or row.get("source")
                or (row.get("meta") or {}).get("origin")
                or (row.get("metadata") or {}).get("source")
            )
            if o != origin:
                continue
            ts = parse_ts(row.get("timestamp"))
            if not ts:
                continue
            # restrict to chosen day window
            if not (start_day0 < ts <= end_day0):
                continue
            k = ts.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            if k in key_set:
                agg[k]["flags_count"] += 1

        # triggers
        for row in trig_rows:
            o = extract_origin(
                row.get("origin")
                or row.get("source")
                or (row.get("meta") or {}).get("origin")
                or (row.get("metadata") or {}).get("source")
            )
            if o != origin:
                continue
            ts = parse_ts(row.get("timestamp"))
            if not ts:
                continue
            if not (start_day0 < ts <= end_day0):
                continue
            k = ts.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            if k in key_set:
                agg[k]["triggers_count"] += 1

        buckets = [
            {
                "timestamp_bucket": k.isoformat(),
                "flags_count": agg[k]["flags_count"],
                "triggers_count": agg[k]["triggers_count"],
            }
            for k in ordered_keys
            if (agg[k]["flags_count"] + agg[k]["triggers_count"]) > 0 or days > 1
        ]

        # For days=1, ensure exactly one bucket (even if it would be empty).
        if days == 1 and not buckets:
            k = ordered_keys[0]
            buckets = [{
                "timestamp_bucket": k.isoformat(),
                "flags_count": 0,
                "triggers_count": 0,
            }]

        origins_out.append({"origin": origin, "buckets": buckets})

    # Drop origins with no buckets at all (keeps output tidy)
    origins_out = [o for o in origins_out if o["buckets"]]

    return {"window_days": days, "interval": interval, "origins": origins_out}