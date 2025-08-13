# src/analytics/origin_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timezone

# --- alias map / normalization ---
_ALIAS = {
    "twitter_api": "twitter",
    "twitterapi": "twitter",
    "twitter": "twitter",
    "reddit_api": "reddit",
    "rss": "rss_news",
    "rssnews": "rss_news",
    "rss_news": "rss_news",
    "market": "market_feed",
    "marketfeed": "market_feed",
    "market_feed": "market_feed",
}

def normalize_origin(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    key = str(raw).strip().lower()
    return _ALIAS.get(key, key)

# --- timestamp parsing that accepts epoch or ISO8601 ---
def _parse_timestamp(ts_val: Any) -> Optional[float]:
    """Return POSIX seconds (float) or None if ts is unusable."""
    if ts_val is None:
        return None
    # already numeric?
    try:
        return float(ts_val)
    except (TypeError, ValueError):
        pass
    # ISO formats (with or without Z)
    try:
        s = str(ts_val)
        # Basic tolerance for trailing Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.timestamp()
    except Exception:
        return None

def _stream_jsonl(path: Path):
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
                # skip malformed lines
                continue

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int,
    include_triggers: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns (rows, totals)
      rows: [{"origin": str, "count": int, "pct": float}, ...] sorted desc by count, asc by origin
      totals: {"flags": int, "triggers": int, "total_events": int}
    """
    if days <= 0:
        # caller should 400, but guard anyway
        return ([], {"flags": 0, "triggers": 0, "total_events": 0})

    now = datetime.now(timezone.utc).timestamp()
    cutoff = now - days * 86400

    # Count flags per origin
    flags_by_origin: Dict[str, int] = {}
    flags_total = 0
    for rec in _stream_jsonl(flags_path):
        ts = _parse_timestamp(rec.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        origin = normalize_origin(rec.get("origin"))
        flags_by_origin[origin] = flags_by_origin.get(origin, 0) + 1
        flags_total += 1

    # Optionally count triggers per origin
    triggers_by_origin: Dict[str, int] = {}
    triggers_total = 0
    if include_triggers:
        for rec in _stream_jsonl(triggers_path):
            ts = _parse_timestamp(rec.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            origin = normalize_origin(rec.get("origin"))
            triggers_by_origin[origin] = triggers_by_origin.get(origin, 0) + 1
            triggers_total += 1

    # Merge for rows (but keep independent totals)
    all_origins = set(flags_by_origin) | set(triggers_by_origin)
    total_events = flags_total + (triggers_total if include_triggers else 0)

    rows: List[Dict[str, Any]] = []
    if total_events > 0:
        for origin in all_origins:
            c = flags_by_origin.get(origin, 0) + (triggers_by_origin.get(origin, 0) if include_triggers else 0)
            pct = round(100.0 * c / total_events, 2)
            rows.append({"origin": origin, "count": c, "pct": pct})

        # Sort: count desc, origin asc
        rows.sort(key=lambda r: (-r["count"], r["origin"]))
    else:
        # Nothing in window
        rows = []

    totals = {
        "flags": flags_total,
        "triggers": triggers_total if include_triggers else 0,
        "total_events": total_events,
    }
    return rows, totals