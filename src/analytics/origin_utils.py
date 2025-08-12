# src/analytics/origin_utils.py

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

__all__ = ["compute_origin_breakdown", "normalize_origin"]

# ---------- parsing & normalization helpers ----------

def _iter_jsonl(path: Path) -> Iterable[dict]:
    """Stream-parse a JSONL file, yielding dicts; skip malformed lines."""
    if not path or not Path(path).exists():
        return
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                # tolerate malformed lines
                continue


def _parse_ts(ts_val) -> Optional[datetime]:
    """Accept unix seconds (int/float) or ISO 8601; return aware UTC dt or None."""
    if ts_val is None:
        return None
    try:
        # numeric epoch
        return datetime.fromtimestamp(float(ts_val), tz=timezone.utc)
    except Exception:
        pass
    # ISO-like string
    try:
        s = str(ts_val)
        # best-effort: add Z → +00:00 if present
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


# Aliases → canonical names
_ORIGIN_MAP: Dict[str, str] = {
    "twitter_api": "twitter",
    "Twitter": "twitter",
    "TWITTER": "twitter",
    "rss": "rss_news",
    "rss-news": "rss_news",
    "rss_news": "rss_news",
    "reddit_api": "reddit",
    "Reddit": "reddit",
    "market": "market_feed",
    "Market": "market_feed",
}


def normalize_origin(value) -> str:
    """Lowercase + alias map; default to 'unknown' when empty."""
    if value is None:
        return "unknown"
    s = str(value).strip()
    if not s:
        return "unknown"
    s_lower = s.lower()
    # First: handle simple lowercase aliases
    if s_lower in _ORIGIN_MAP:
        return _ORIGIN_MAP[s_lower]
    # Second: map original (for mixed case keys present in map)
    return _ORIGIN_MAP.get(s, s_lower)


# ---------- core computation ----------

def _tally_events(
    path: Path,
    cutoff_utc: datetime,
) -> Tuple[Counter, int]:
    """
    Count events per normalized origin for a single file (flags OR triggers).
    Returns (origin_counter, total_count_in_window).
    """
    counts: Counter = Counter()
    total = 0
    for obj in _iter_jsonl(path):
        dt = _parse_ts(obj.get("timestamp"))
        if not dt or dt < cutoff_utc:
            continue
        origin = normalize_origin(obj.get("origin"))
        counts[origin] += 1
        total += 1
    return counts, total


def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Optional[Path],
    days: int,
    include_triggers: bool,
):
    """
    Returns (origins_list, totals_dict)

    origins_list: [
      {"origin": "twitter", "count": 64, "pct": 52.03},
      ...
    ]

    totals_dict: {"flags": int, "triggers": int, "total_events": int}
    """
    # Cutoff
    now_utc = datetime.now(timezone.utc)
    cutoff_utc = now_utc - timedelta(days=days)

    # Flags
    flag_counts, flags_total = _tally_events(flags_path, cutoff_utc)

    # Triggers (optional)
    trig_counts: Counter = Counter()
    triggers_total = 0
    if include_triggers and triggers_path:
        trig_counts, triggers_total = _tally_events(triggers_path, cutoff_utc)

    # Merge counts for output percentages
    merged: Counter = flag_counts.copy()
    merged.update(trig_counts)

    total_events = flags_total + (triggers_total if include_triggers else 0)

    # Build list with percentages
    if total_events == 0:
        origins_list = []
    else:
        origins_list = [
            {
                "origin": origin,
                "count": count,
                "pct": round(100.0 * count / total_events, 2),
            }
            for origin, count in merged.items()
        ]
        # sort by count desc then origin asc
        origins_list.sort(key=lambda x: (-x["count"], x["origin"]))

    totals = {
        "flags": flags_total,
        "triggers": triggers_total if include_triggers else 0,
        "total_events": total_events,
    }
    return origins_list, totals