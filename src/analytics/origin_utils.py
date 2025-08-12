# src/analytics/origin_utils.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Any
from datetime import datetime, timedelta, timezone

# ---- Normalization -----------------------------------------------------------

_ALIAS_MAP = {
    "twitter_api": "twitter",
    "twitterapi": "twitter",
    "x": "twitter",
    "rss": "rss_news",
    "rssfeed": "rss_news",
    "news_rss": "rss_news",
}

def normalize_origin(origin: Any) -> str:
    """Lowercase + alias map; fallback to 'unknown' if missing/empty."""
    if origin is None:
        return "unknown"
    s = str(origin).strip().lower()
    if not s:
        return "unknown"
    return _ALIAS_MAP.get(s, s)

# ---- IO helpers --------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterable[dict]:
    """Stream a JSONL file; tolerate malformed lines."""
    if not path or not Path(path).exists():
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield obj

def _parse_timestamp(ts: Any) -> float | None:
    """Accept epoch seconds (int/float/str), or ISO 8601. Return epoch seconds UTC."""
    if ts is None:
        return None
    # epoch-like
    try:
        return float(ts)
    except Exception:
        pass
    # ISO 8601
    try:
        s = str(ts)
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

# ---- Core computation --------------------------------------------------------

def _count_events_since(path: Path, since_ts: float) -> Tuple[Dict[str, int], int]:
    """
    Count events per normalized origin for records whose timestamp >= since_ts.
    Returns (counts_by_origin, total_count).
    """
    counts: Dict[str, int] = {}
    total = 0
    for rec in _iter_jsonl(path):
        ts = _parse_timestamp(rec.get("timestamp"))
        if ts is None or ts < since_ts:
            continue
        origin = normalize_origin(rec.get("origin"))
        counts[origin] = counts.get(origin, 0) + 1
        total += 1
    return counts, total

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int = 7,
    include_triggers: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Compute origin breakdown.

    Returns:
      rows: List[{origin, count, pct}]
      totals: {"flags": int, "triggers": int, "total_events": int}
    """
    if days <= 0:
        # Let caller validate; we keep this pure
        return [], {"flags": 0, "triggers": 0, "total_events": 0}

    now = datetime.now(timezone.utc).timestamp()
    since_ts = now - days * 86400

    # Flags
    flag_counts, flags_total = _count_events_since(flags_path, since_ts)

    # Triggers (optional)
    trigger_counts: Dict[str, int] = {}
    triggers_total = 0
    if include_triggers:
        trigger_counts, triggers_total = _count_events_since(triggers_path, since_ts)

    # Merge for output counts
    merged: Dict[str, int] = dict(flag_counts)  # copy
    if include_triggers:
        for k, v in trigger_counts.items():
            merged[k] = merged.get(k, 0) + v

    total_events = flags_total + (triggers_total if include_triggers else 0)

    # Build rows with percentages; handle total_events == 0
    rows: List[Dict[str, Any]] = []
    if total_events > 0:
        for origin, cnt in merged.items():
            pct = round(100.0 * cnt / total_events, 2)
            rows.append({"origin": origin, "count": cnt, "pct": pct})
        # sort: count desc, origin asc
        rows.sort(key=lambda r: (-r["count"], r["origin"]))
    else:
        rows = []

    totals = {
        "flags": flags_total,
        "triggers": triggers_total if include_triggers else 0,
        "total_events": total_events,
    }
    return rows, totals