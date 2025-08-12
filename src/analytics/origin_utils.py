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
    p = Path(path) if path is not None else None
    if not p or not p.exists():
        return
    with p.open("r") as f:
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
        days = 1

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    # Per-origin counts across included streams
    by_origin: Dict[str, int] = {}

    # Keep separate tallies
    flags_count = 0
    triggers_count = 0

    # Deduplicate per stream to avoid accidental duplicate log lines
    seen_flags: set[tuple[str, str, float]] = set()
    seen_trigs: set[tuple[str, str, float]] = set()

    # ---- flags ----
    for rec in _stream_jsonl(flags_path):
        ts_dt = _parse_ts(rec.get("timestamp"))
        if ts_dt is None or ts_dt < cutoff:
            continue
        origin = normalize_origin(rec.get("origin"))
        signal_id = str(rec.get("signal_id", ""))
        ts_key = ts_dt.timestamp()
        key = (signal_id, origin, ts_key)
        if key in seen_flags:
            continue
        seen_flags.add(key)

        by_origin[origin] = by_origin.get(origin, 0) + 1
        flags_count += 1

    # ---- triggers ----
    if include_triggers:
        for rec in _stream_jsonl(triggers_path):
            ts_dt = _parse_ts(rec.get("timestamp"))
            if ts_dt is None or ts_dt < cutoff:
                continue
            origin = normalize_origin(rec.get("origin"))
            signal_id = str(rec.get("signal_id", ""))
            ts_key = ts_dt.timestamp()
            key = (signal_id, origin, ts_key)
            if key in seen_trigs:
                continue
            seen_trigs.add(key)

            by_origin[origin] = by_origin.get(origin, 0) + 1
            triggers_count += 1

    total_events = flags_count + (triggers_count if include_triggers else 0)

    # Build rows
    rows: List[Dict[str, Any]] = []
    for origin, count in by_origin.items():
        pct = (100.0 * count / total_events) if total_events > 0 else 0.0
        rows.append({
            "origin": origin,
            "count": count,
            "pct": round(pct, 2),
        })

    # Order: count desc, then origin asc
    rows.sort(key=lambda r: (-r["count"], r["origin"]))

    totals = {
        "flags": flags_count,
        "triggers": triggers_count if include_triggers else 0,
        "total_events": total_events,
    }
    return rows, totals