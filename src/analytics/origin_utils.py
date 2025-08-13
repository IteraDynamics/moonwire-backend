# src/analytics/origin_utils.py

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# --- origin normalization ---

_ALIAS_MAP = {
    "twitter_api": "twitter",
    "Twitter": "twitter",
    "reddit_api": "reddit",
    "rss": "rss_news",
    "news": "rss_news",
}

def normalize_origin(origin: object) -> str:
    if origin is None:
        return "unknown"
    s = str(origin).strip()
    if not s:
        return "unknown"
    # first pass through alias map (case sensitive for known variants)
    s = _ALIAS_MAP.get(s, s)
    # then lowercase (keeps already mapped stable)
    s = s.lower()
    # map again in case the lowercase hits an alias key
    return _ALIAS_MAP.get(s, s)


# --- parsing & window filter ---

def _iter_jsonl(path: Path) -> Iterable[dict]:
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

def _ts_in_window(ts_val: object, window_start: float) -> bool:
    """
    Accepts numeric epoch seconds or ISO8601 strings (with/without Z).
    """
    if ts_val is None:
        return False
    # numeric?
    try:
        ts = float(ts_val)
        return ts >= window_start
    except Exception:
        pass
    # iso?
    try:
        s = str(ts_val)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.timestamp() >= window_start
    except Exception:
        return False


# --- main API ---

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int,
    include_triggers: bool,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Returns (rows, totals) where:
      rows: [{"origin": str, "count": int, "pct": float}, ...] sorted by count desc, origin asc
      totals: {"flags": int, "triggers": int, "total_events": int}
    """

    if days <= 0:
        # let the router handle 400; keep util pure/simple
        return [], {"flags": 0, "triggers": 0, "total_events": 0}

    now = datetime.now(timezone.utc).timestamp()
    window_start = now - days * 86400

    # --- collect flags (only from flags_path) ---
    flag_events: List[dict] = []
    for rec in _iter_jsonl(flags_path):
        if _ts_in_window(rec.get("timestamp"), window_start):
            flag_events.append(rec)

    flags_count = len(flag_events)

    # --- collect triggers if requested (only from triggers_path) ---
    trigger_events: List[dict] = []
    if include_triggers:
        for rec in _iter_jsonl(triggers_path):
            if _ts_in_window(rec.get("timestamp"), window_start):
                trigger_events.append(rec)

    triggers_count = len(trigger_events)

    # --- aggregate by origin over the combined set ---
    combined = flag_events + trigger_events
    by_origin: Dict[str, int] = defaultdict(int)
    for rec in combined:
        origin = normalize_origin(rec.get("origin"))
        by_origin[origin] += 1

    total_events = flags_count + triggers_count
    # Build rows with pct (guard divide-by-zero)
    rows: List[Dict] = []
    for origin, count in by_origin.items():
        pct = round(100.0 * count / total_events, 2) if total_events > 0 else 0.0
        rows.append({"origin": origin, "count": count, "pct": pct})

    # sort: count desc, then origin asc
    rows.sort(key=lambda r: (-r["count"], r["origin"]))

    totals = {
        "flags": flags_count,
        "triggers": triggers_count,
        "total_events": total_events,
    }
    return rows, totals