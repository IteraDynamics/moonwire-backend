# src/analytics/origin_utils.py

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple, Any


# ---- timestamp parsing -------------------------------------------------------

def _parse_ts(val: Any) -> datetime | None:
    """Accept unix seconds (int/float) or ISO8601; return tz-aware UTC or None."""
    if val is None:
        return None
    # unix epoch?
    try:
        ts = float(val)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    # ISO8601
    try:
        s = str(val)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path or not Path(path).exists():
        return []
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip malformed line
                continue


# ---- origin normalization ----------------------------------------------------

_ALIAS = {
    "twitter_api": "twitter",
    "twitterapi": "twitter",
    "Twitter": "twitter",
    "rss": "rss_news",
    "rss-news": "rss_news",
}

def _normalize_origin(o: Any) -> str:
    if not o:
        return "unknown"
    s = str(o).strip()
    if not s:
        return "unknown"
    key = s.lower()
    return _ALIAS.get(s, _ALIAS.get(key, key))


# ---- public API --------------------------------------------------------------

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int = 7,
    include_triggers: bool = True,
) -> Tuple[list[dict], dict]:
    """
    Stream logs and compute counts per origin.

    Returns:
      rows: [{"origin": ..., "count": int, "pct": float}, ...] sorted
      totals: {"total_events": int, "flags": int, "triggers": int}
    """
    if days <= 0:
        return [], {"total_events": 0, "flags": 0, "triggers": 0}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    counts: Dict[str, int] = defaultdict(int)
    flags_count = 0
    triggers_count = 0

    # Always count flags
    for rec in _iter_jsonl(flags_path):
        ts = _parse_ts(rec.get("timestamp"))
        if not ts or ts < cutoff:
            continue
        origin = _normalize_origin(rec.get("origin"))
        counts[origin] += 1
        flags_count += 1

    # Optionally add triggers
    if include_triggers:
        for rec in _iter_jsonl(triggers_path):
            ts = _parse_ts(rec.get("timestamp"))
            if not ts or ts < cutoff:
                continue
            origin = _normalize_origin(rec.get("origin"))
            counts[origin] += 1
            triggers_count += 1

    total_events = flags_count + (triggers_count if include_triggers else 0)

    # Build rows (pct of *included* total)
    rows = []
    for origin, cnt in counts.items():
        pct = round((cnt / total_events * 100.0), 2) if total_events > 0 else 0.0
        rows.append({"origin": origin, "count": cnt, "pct": pct})

    # Sort: count desc, then origin asc
    rows.sort(key=lambda r: (-r["count"], r["origin"]))

    totals = {
        "total_events": total_events,
        "flags": flags_count,
        "triggers": triggers_count if include_triggers else 0,
    }
    return rows, totals