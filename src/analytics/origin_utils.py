# src/analytics/origin_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime, timedelta, timezone

# --- helpers ------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _parse_ts(ts_raw) -> datetime | None:
    """
    Accepts float/int epoch seconds or ISO8601 string (with/without Z).
    Returns aware UTC datetime or None if unparsable.
    """
    if ts_raw is None:
        return None
    # numeric epoch
    try:
        return datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
    except Exception:
        pass
    # ISO
    try:
        s = str(ts_raw)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

_ALIAS_MAP = {
    "twitter_api": "twitter",
    "Twitter": "twitter",
    "reddit_api": "reddit",
    "rss": "rss_news",
}

def _norm_origin(origin) -> str:
    if origin is None:
        return "unknown"
    key = str(origin).strip()
    if not key:
        return "unknown"
    return _ALIAS_MAP.get(key, key.lower())

def _read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip malformed
                continue

# --- core ---------------------------------------------------------------

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    days: int = 7,
    include_triggers: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int | float]]:
    """
    Stream both logs and return (rows, totals).

    rows: list of {origin, count, pct}
    totals: {"flags": int, "triggers": int, "total_events": int}
    """
    if days <= 0:
        raise ValueError("days must be >= 1")

    cutoff = _now_utc() - timedelta(days=days)

    origin_counts: Dict[str, int] = {}
    flag_count = 0
    trig_count = 0

    # Count FLAGS
    for row in _read_jsonl(flags_path):
        ts = _parse_ts(row.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        origin = _norm_origin(row.get("origin"))
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
        flag_count += 1

    # Count TRIGGERS (optionally)
    if include_triggers:
        for row in _read_jsonl(triggers_path):
            ts = _parse_ts(row.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            origin = _norm_origin(row.get("origin"))
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
            trig_count += 1

    total_events = flag_count + (trig_count if include_triggers else 0)

    # Build rows with pct
    rows = []
    if total_events > 0:
        for origin, count in origin_counts.items():
            pct = round(100.0 * count / total_events, 2)
            rows.append({"origin": origin, "count": count, "pct": pct})
        # sort by count desc, then origin asc for stability
        rows.sort(key=lambda x: (-x["count"], x["origin"]))
    # if no events, return empty rows

    totals = {
        "flags": flag_count,
        "triggers": (trig_count if include_triggers else 0),
        "total_events": total_events,
    }
    return rows, totals