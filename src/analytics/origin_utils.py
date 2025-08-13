# src/analytics/origin_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple, Any, List
from datetime import datetime, timedelta, timezone

# ---------- Helpers ----------

_ALIAS_MAP = {
    "twitter_api": "twitter",
    "twitterapi": "twitter",
    "twitter": "twitter",
    "reddit": "reddit",
    "rss": "rss_news",
    "rss_news": "rss_news",
    "market": "market_feed",
    "market_feed": "market_feed",
}

_ORIGIN_FIELDS = ("origin", "source", "provider", "channel")


def normalize_origin(raw: Any) -> str:
    if raw is None:
        return "unknown"
    try:
        s = str(raw).strip().lower()
    except Exception:
        return "unknown"
    return _ALIAS_MAP.get(s, s if s else "unknown")


def parse_timestamp(val: Any) -> datetime | None:
    """
    Accept epoch seconds (int/float/str) or ISO8601 (with/without Z/offset).
    Return timezone-aware UTC datetime or None if unparsable.
    """
    if val is None:
        return None

    # Epoch-like
    try:
        # common case: float/int or numeric string
        ts = float(val)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass

    # ISO-like
    try:
        s = str(val)
        # naive or 'Z' → assume UTC
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                # ignore malformed lines
                continue


def _extract_origin(row: dict) -> str:
    for key in _ORIGIN_FIELDS:
        if key in row:
            return normalize_origin(row.get(key))
    return "unknown"


def aggregate_counts(
    flags_path: Path,
    triggers_path: Path | None,
    *,
    days: int,
    include_triggers: bool,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Returns (flag_counts, trigger_counts, included_totals)
      - flag_counts: origin -> count (flags only)
      - trigger_counts: origin -> count (triggers only, possibly empty)
      - included_totals: {"flags": int, "triggers": int, "total_events": int}
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    flag_counts: Dict[str, int] = {}
    trigger_counts: Dict[str, int] = {}

    # Every line in flags_path is a "flag" event.
    for row in _iter_jsonl(flags_path):
        ts = parse_timestamp(row.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        origin = _extract_origin(row)
        flag_counts[origin] = flag_counts.get(origin, 0) + 1

    flags_total = sum(flag_counts.values())

    triggers_total = 0
    if include_triggers and triggers_path is not None:
        for row in _iter_jsonl(triggers_path):
            ts = parse_timestamp(row.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            origin = _extract_origin(row)
            trigger_counts[origin] = trigger_counts.get(origin, 0) + 1
        triggers_total = sum(trigger_counts.values())

    totals = {
        "flags": flags_total,
        "triggers": triggers_total,
        "total_events": flags_total + (triggers_total if include_triggers else 0),
    }
    return flag_counts, trigger_counts, totals


def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int,
    include_triggers: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Builds a sorted list of {origin, count, pct} and a totals dict.

    Sorting: count desc, then origin asc.
    pct is based on total_events (flags + triggers if included).
    """
    flag_counts, trigger_counts, totals = aggregate_counts(
        flags_path, triggers_path, days=days, include_triggers=include_triggers
    )

    # Merge counts if including triggers; otherwise flags only
    combined: Dict[str, int] = dict(flag_counts)
    if include_triggers:
        for k, v in trigger_counts.items():
            combined[k] = combined.get(k, 0) + v

    total_events = totals["total_events"]
    rows: List[Dict[str, Any]] = []
    if total_events > 0:
        for origin, cnt in combined.items():
            pct = round(100.0 * cnt / total_events, 2)
            rows.append({"origin": origin, "count": cnt, "pct": pct})
    else:
        # no events; return empty rows and zeros in totals
        rows = []

    # Sort by count desc, then origin asc
    rows.sort(key=lambda r: (-r["count"], r["origin"]))
    return rows, totals