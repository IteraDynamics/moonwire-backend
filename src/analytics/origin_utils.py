# src/analytics/origin_utils.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone


# ---- origin normalization -----------------------------------------------------

_ORIGIN_ALIASES = {
    "twitter_api": "twitter",
    "Twitter": "twitter",
    "rss": "rss_news",
    "RSS": "rss_news",
    "reddit_api": "reddit",
}

def normalize_origin(raw: Any) -> str:
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    if not s:
        return "unknown"
    return _ORIGIN_ALIASES.get(s, _ORIGIN_ALIASES.get(s.lower(), s.lower()))


# ---- timestamp parsing --------------------------------------------------------

def _parse_ts(v: Any) -> datetime | None:
    """
    Accepts:
      - float/int epoch seconds
      - ISO 8601 string (with/without Z)
    Returns aware UTC datetime or None if unparsable.
    """
    if v is None:
        return None
    # epoch numeric?
    try:
        ts = float(v)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    # ISO string
    try:
        s = str(v)
        # fast path for "....Z"
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        # make aware in UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


# ---- streaming JSONL reader ---------------------------------------------------

def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip malformed to keep endpoint resilient
                continue


# ---- main aggregation ---------------------------------------------------------

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int,
    include_triggers: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns (rows, totals)
      rows: [{origin, count, pct}, ...] sorted by count desc, origin asc
      totals: {"flags": int, "triggers": int, "total_events": int}
    """

    if days <= 0:
        # caller should validate, but be defensive
        return [], {"flags": 0, "triggers": 0, "total_events": 0}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    counts: Dict[str, int] = {}
    flags_count = 0
    triggers_count = 0

    # ---- flags (always included) ----
    for rec in _iter_jsonl(flags_path):
        ts = _parse_ts(rec.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        origin = normalize_origin(rec.get("origin"))
        counts[origin] = counts.get(origin, 0) + 1
        flags_count += 1

    # ---- triggers (conditionally included) ----
    if include_triggers:
        for rec in _iter_jsonl(triggers_path):
            ts = _parse_ts(rec.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            origin = normalize_origin(rec.get("origin"))
            counts[origin] = counts.get(origin, 0) + 1
            triggers_count += 1

    total_events = flags_count + (triggers_count if include_triggers else 0)

    # Build rows with pct
    rows: List[Dict[str, Any]] = []
    if total_events > 0:
        for origin, count in counts.items():
            pct = round(100.0 * count / total_events, 2)
            rows.append({"origin": origin, "count": count, "pct": pct})
        # sort by count desc, then origin asc
        rows.sort(key=lambda r: (-r["count"], r["origin"]))
    else:
        rows = []

    totals = {
        "flags": flags_count,
        "triggers": (triggers_count if include_triggers else 0),
        "total_events": total_events,
    }
    return rows, totals