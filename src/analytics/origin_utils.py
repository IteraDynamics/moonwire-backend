# src/analytics/origin_utils.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta, timezone

# --- origin normalization ---

_ALIAS_MAP = {
    "twitter_api": "twitter",
    "twitter": "twitter",
    "x": "twitter",
    "rss": "rss_news",
    "rss-news": "rss_news",
    "reddit_api": "reddit",
}

def normalize_origin(value: Any) -> str:
    if value is None:
        return "unknown"
    s = str(value).strip().lower()
    if not s:
        return "unknown"
    return _ALIAS_MAP.get(s, s)

# --- time parsing ---

def _parse_ts(v: Any) -> datetime | None:
    if v is None:
        return None
    # epoch seconds (int/float/str)
    try:
        ts = float(v)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    # ISO 8601
    try:
        s = str(v)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

# --- JSONL streaming ---

def _stream_jsonl(path: Path):
    if not path or not Path(path).exists():
        return
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # tolerate malformed lines
                continue

# --- main aggregation ---

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    days: int = 7,
    include_triggers: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns (rows, totals) where:
      rows: [{origin, count, pct}, ...] sorted by count desc, origin asc
      totals: {"flags": int, "triggers": int, "total_events": int}
    """
    if days < 1:
        # router validates, but keep defensive
        days = 1

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    # Per-origin counts across whichever streams are included
    by_origin: Dict[str, int] = {}
    flags_count = 0
    triggers_count = 0

    # Count flags
    for rec in _stream_jsonl(flags_path):
        ts = _parse_ts(rec.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        origin = normalize_origin(rec.get("origin"))
        by_origin[origin] = by_origin.get(origin, 0) + 1
        flags_count += 1

    # Count triggers (only if requested)
    if include_triggers:
        for rec in _stream_jsonl(triggers_path):
            ts = _parse_ts(rec.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            origin = normalize_origin(rec.get("origin"))
            by_origin[origin] = by_origin.get(origin, 0) + 1
            triggers_count += 1

    total_events = flags_count + (triggers_count if include_triggers else 0)

    rows: List[Dict[str, Any]] = []
    for origin, count in by_origin.items():
        pct = (100.0 * count / total_events) if total_events > 0 else 0.0
        rows.append({
            "origin": origin,
            "count": count,
            "pct": round(pct, 2),
        })

    # Sort: count desc, then origin asc
    rows.sort(key=lambda r: (-r["count"], r["origin"]))

    totals = {
        "flags": flags_count,
        "triggers": triggers_count if include_triggers else 0,
        "total_events": total_events,
    }
    return rows, totals