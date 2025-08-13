# src/analytics/origin_utils.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Iterable
from datetime import datetime, timezone

# --- alias map / normalization ---
_ALIAS = {
    "twitter_api": "twitter",
    "twitterapi":  "twitter",
    "twitter":     "twitter",
    "x":           "twitter",
    "reddit_api":  "reddit",
    "reddit":      "reddit",
    "rss":         "rss_news",
    "rssnews":     "rss_news",
    "rss_news":    "rss_news",
    "news":        "rss_news",
    "market":      "market_feed",
    "marketfeed":  "market_feed",
    "market_feed": "market_feed",
}

def _normalize_origin(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    low = str(raw).strip().lower()
    return _ALIAS.get(low, low)

def _extract_origin(rec: dict) -> str:
    """
    Tolerate schema variants:
      - origin / source / signal_origin (top-level)
      - meta.origin / metadata.source (nested)
    """
    o = rec.get("origin") or rec.get("source") or rec.get("signal_origin")
    if not o:
        meta = rec.get("meta") or rec.get("metadata") or {}
        o = meta.get("origin") or meta.get("source")
    return _normalize_origin(o)

# --- timestamp parsing that accepts epoch or ISO8601 (with/without Z) ---
def _parse_timestamp(ts_val: Any) -> Optional[float]:
    """Return POSIX seconds (float) or None if unusable."""
    if ts_val is None:
        return None
    # numeric or numeric-like string
    try:
        return float(ts_val)
    except (TypeError, ValueError):
        pass
    # ISO formats (with or without Z / timezone)
    try:
        s = str(ts_val)
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

def _stream_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
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
        return ([], {"flags": 0, "triggers": 0, "total_events": 0})

    now_ts = time.time()
    cutoff = now_ts - days * 86400

    # Count flags per origin
    flags_by_origin: Dict[str, int] = {}
    flags_total = 0
    for rec in _stream_jsonl(flags_path):
        ts = _parse_timestamp(rec.get("timestamp"))
        # Treat missing/unparsable timestamp as "now" (internal analytics tolerance)
        if ts is None:
            ts = now_ts
        if ts < cutoff:
            continue
        origin = _extract_origin(rec)
        flags_by_origin[origin] = flags_by_origin.get(origin, 0) + 1
        flags_total += 1

    # Optionally count triggers per origin
    triggers_by_origin: Dict[str, int] = {}
    triggers_total = 0
    if include_triggers:
        for rec in _stream_jsonl(triggers_path):
            ts = _parse_timestamp(rec.get("timestamp"))
            if ts is None:
                ts = now_ts
            if ts < cutoff:
                continue
            origin = _extract_origin(rec)
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
        rows.sort(key=lambda r: (-r["count"], r["origin"]))

    totals = {
        "flags": flags_total,
        "triggers": triggers_total if include_triggers else 0,
        "total_events": total_events,
    }
    return rows, totals
