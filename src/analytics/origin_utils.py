# src/analytics/origin_utils.py

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple


# ----------------------- parsing & helpers -----------------------

def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Stream a JSONL file line-by-line, yielding dicts.
    Tolerates missing files and malformed lines.
    """
    if not path or not isinstance(path, Path) or not path.exists():
        return
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    yield rec
            except Exception:
                # swallow malformed lines
                continue


def _parse_ts(ts_val: Any) -> datetime | None:
    """
    Accepts float/ints (epoch seconds) or ISO 8601 strings.
    Returns aware UTC datetime, or None if unparseable.
    """
    if ts_val is None:
        return None

    # epoch seconds (int/float or stringified)
    try:
        tsf = float(ts_val)
        return datetime.fromtimestamp(tsf, tz=timezone.utc)
    except Exception:
        pass

    # ISO strings; make 'Z' explicit UTC if present
    try:
        s = str(ts_val)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _ts_in_window(ts_val: Any, window_start: datetime) -> bool:
    dt = _parse_ts(ts_val)
    if dt is None:
        return False
    return dt >= window_start


# ----------------------- origin normalization -----------------------

_ALIAS_MAP = {
    # twitter
    "twitter_api": "twitter",
    "twitterapi": "twitter",
    "twitter": "twitter",
    "x": "twitter",
    # reddit
    "reddit_api": "reddit",
    "redditapi": "reddit",
    "reddit": "reddit",
    # news / rss
    "rss": "rss_news",
    "rssnews": "rss_news",
    "rss_news": "rss_news",
    "news": "rss_news",
    "rss-feed": "rss_news",
    # markets
    "market": "market_feed",
    "markets": "market_feed",
    "market_feed": "market_feed",
    "marketfeed": "market_feed",
}

def normalize_origin(origin: Any) -> str:
    if origin is None:
        return "unknown"
    key = str(origin).strip()
    if not key:
        return "unknown"
    low = key.lower()
    return _ALIAS_MAP.get(low, low)


# ----------------------- main aggregation -----------------------

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int = 7,
    include_triggers: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Compute per-origin counts within a lookback window and return:
      - rows: [{origin, count, pct}, ...] sorted by count desc then origin asc
      - totals: {"flags": <unique signals>, "triggers": <unique signals>, "total_events": sum}

    NOTE:
    - For 'included' tallies, we count UNIQUE signal_ids (what tests expect).
    - For per-origin 'count', we aggregate per EVENT line (flags + triggers if included).
    - Timestamps can be epoch seconds or ISO strings; unparsable lines are ignored.
    """
    now = datetime.now(timezone.utc)
    if days <= 0:
        # router should validate, but keep util safe
        days = 1
    window_start = now - timedelta(days=days)

    # ---- collect flags (from flags_path only) ----
    flag_events: List[Dict[str, Any]] = []
    for rec in _iter_jsonl(flags_path):
        if _ts_in_window(rec.get("timestamp"), window_start):
            flag_events.append(rec)

    # Unique signal IDs for flags tallies
    flag_ids = {rec.get("signal_id") for rec in flag_events if rec.get("signal_id") is not None}
    flags_count = len(flag_ids)

    # ---- collect triggers if requested (from triggers_path only) ----
    trigger_events: List[Dict[str, Any]] = []
    if include_triggers:
        for rec in _iter_jsonl(triggers_path):
            if _ts_in_window(rec.get("timestamp"), window_start):
                trigger_events.append(rec)

    # Unique signal IDs for triggers tallies
    trigger_ids = {rec.get("signal_id") for rec in trigger_events if rec.get("signal_id") is not None}
    triggers_count = len(trigger_ids)

    # ---- aggregate by origin (per event) ----
    combined: List[Dict[str, Any]] = flag_events + (trigger_events if include_triggers else [])
    by_origin: Dict[str, int] = defaultdict(int)
    for rec in combined:
        origin = normalize_origin(rec.get("origin"))
        by_origin[origin] += 1

    total_events = flags_count + triggers_count

    # Build rows with pct relative to included totals
    rows: List[Dict[str, Any]] = []
    if total_events > 0:
        for origin, count in by_origin.items():
            pct = round(100.0 * count / float(total_events), 2)
            rows.append({"origin": origin, "count": count, "pct": pct})
        # sort: count desc, then origin asc
        rows.sort(key=lambda r: (-r["count"], r["origin"]))
    else:
        rows = []

    totals = {
        "flags": flags_count,
        "triggers": triggers_count,
        "total_events": total_events,
    }
    return rows, totals