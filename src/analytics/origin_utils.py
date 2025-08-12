# src/analytics/origin_utils.py

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any


# --- helpers -----------------------------------------------------------------

def _parse_ts(val: Any) -> datetime | None:
    """
    Accepts float/ints epoch seconds or ISO8601 strings (with or without 'Z').
    Returns aware UTC datetime or None if unparsable.
    """
    if val is None:
        return None
    # epoch seconds?
    try:
        return datetime.fromtimestamp(float(val), tz=timezone.utc)
    except Exception:
        pass

    # ISO8601?
    try:
        s = str(val)
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


_ALIAS = {
    "twitter_api": "twitter",
    "Twitter": "twitter",
    "rss": "rss_news",
    "RSS": "rss_news",
    "reddit_api": "reddit",
}


def normalize_origin(origin: Any) -> str:
    if not origin:
        return "unknown"
    o = str(origin).strip()
    if not o:
        return "unknown"
    return _ALIAS.get(o, _ALIAS.get(o.lower(), o.lower()))


def _stream_jsonl_in_window(path: Path, cutoff: datetime) -> List[dict]:
    """
    Stream a JSONL file and return only entries with timestamp >= cutoff.
    Skips malformed lines or missing timestamps.
    """
    out: List[dict] = []
    if not path.exists():
        return out
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ts = _parse_ts(obj.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            out.append(obj)
    return out


# --- core API ----------------------------------------------------------------

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    days: int,
    include_triggers: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns:
      - list of {origin, count, pct} rows (unsliced; caller may filter)
      - totals dict: {"flags": int, "triggers": int, "total_events": int}

    Rules:
      * Count flags from flags_path.
      * If include_triggers=True, also count triggers from triggers_path.
      * Normalize origins.
      * Window by timestamp >= now - days.
      * Percentage is of total_events (flags + triggers if included).
      * Sorting: by count desc, then origin asc.
    """
    if days < 1:
        # Let caller validate; but keep safe here as well.
        days = 1

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    # Load windowed events
    flags = _stream_jsonl_in_window(flags_path, cutoff)
    triggers = _stream_jsonl_in_window(triggers_path, cutoff) if include_triggers else []

    # Per-origin counts
    per_origin: Dict[str, int] = {}
    flags_per_origin: Dict[str, int] = {}
    triggers_per_origin: Dict[str, int] = {}

    # Count flags
    for obj in flags:
        origin = normalize_origin(obj.get("origin"))
        flags_per_origin[origin] = flags_per_origin.get(origin, 0) + 1

    # Count triggers if included
    if include_triggers:
        for obj in triggers:
            origin = normalize_origin(obj.get("origin"))
            triggers_per_origin[origin] = triggers_per_origin.get(origin, 0) + 1

    # Merge into per_origin depending on include_triggers
    if include_triggers:
        # union of keys
        all_keys = set(flags_per_origin) | set(triggers_per_origin)
        for k in all_keys:
            per_origin[k] = flags_per_origin.get(k, 0) + triggers_per_origin.get(k, 0)
    else:
        per_origin = dict(flags_per_origin)

    total_flags = sum(flags_per_origin.values())
    total_triggers = sum(triggers_per_origin.values()) if include_triggers else 0
    total_events = total_flags + total_triggers

    # Build rows with pct (guard divide-by-zero)
    rows: List[Dict[str, Any]] = []
    if total_events > 0:
        for origin, cnt in per_origin.items():
            pct = round(100.0 * cnt / total_events, 2)
            rows.append({"origin": origin, "count": cnt, "pct": pct})
        # Sort by count desc, then origin asc
        rows.sort(key=lambda x: (-x["count"], x["origin"]))
    else:
        rows = []

    totals = {
        "flags": total_flags,
        "triggers": total_triggers,
        "total_events": total_events,
    }
    return rows, totals