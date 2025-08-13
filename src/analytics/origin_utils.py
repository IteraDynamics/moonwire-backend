# src/analytics/origin_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Iterable
from datetime import datetime, timezone, timedelta

# ---- origin normalization ----
_ALIAS_MAP = {
    "twitter_api": "twitter",
    "Twitter": "twitter",
    "tweet": "twitter",
    "rss": "rss_news",
    "news": "rss_news",
}

def normalize_origin(raw) -> str:
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    if not s:
        return "unknown"
    key = s.lower()
    return _ALIAS_MAP.get(key, key)

# ---- timestamp parsing ----
def _to_aware_utc(ts_val) -> datetime | None:
    """
    Accept float/int epoch seconds OR ISO8601 string (with/without 'Z').
    Return timezone-aware UTC datetime, or None if unparsable.
    """
    if ts_val is None:
        return None
    # numeric epoch?
    if isinstance(ts_val, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts_val), tz=timezone.utc)
        except Exception:
            return None
    # string: try numeric first, then ISO
    s = str(ts_val).strip()
    if not s:
        return None
    # numeric string?
    try:
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    except Exception:
        pass
    # ISO 8601
    try:
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

# ---- file iterator ----
def _iter_jsonl(path: Path) -> Iterable[dict]:
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

# ---- core computation ----
def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int,
    include_triggers: bool,
) -> Tuple[list[dict], Dict[str, int]]:
    """
    Stream both files, filter to last `days`, normalize origin, count events.
    Returns (rows, totals) where:
      - rows: [{origin, count, pct}] sorted by count desc then origin asc
      - totals: {"flags": int, "triggers": int, "total_events": int}
    """
    if days <= 0:
        # Let the router 400 invalid days; we keep logic simple here
        return [], {"flags": 0, "triggers": 0, "total_events": 0}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    counts: Dict[str, int] = {}

    flags_count = 0
    triggers_count = 0

    # Flags (always included)
    if flags_path and Path(flags_path).exists():
        for rec in _iter_jsonl(flags_path):
            ts = _to_aware_utc(rec.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            origin = normalize_origin(rec.get("origin"))
            counts[origin] = counts.get(origin, 0) + 1
            flags_count += 1

    # Triggers (optional)
    if include_triggers and triggers_path and Path(triggers_path).exists():
        for rec in _iter_jsonl(triggers_path):
            ts = _to_aware_utc(rec.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            origin = normalize_origin(rec.get("origin"))
            counts[origin] = counts.get(origin, 0) + 1
            triggers_count += 1

    total_events = flags_count + (triggers_count if include_triggers else 0)

    # Build rows with pct
    rows = []
    if total_events > 0:
        for origin, cnt in counts.items():
            pct = round(100.0 * cnt / total_events, 2)
            rows.append({"origin": origin, "count": cnt, "pct": pct})
        rows.sort(key=lambda r: (-r["count"], r["origin"]))
    # If total_events == 0, return empty rows

    totals = {
        "flags": flags_count,
        "triggers": triggers_count if include_triggers else 0,
        "total_events": total_events,
    }
    return rows, totals