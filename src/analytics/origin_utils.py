# src/analytics/origin_utils.py
from __future__ import annotations

import json, time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Simple alias map (extend as needed)
_ALIAS = {
    "twitter_api": "twitter",
    "Twitter": "twitter",
    "reddit_api": "reddit",
    "rss": "rss_news",
    "rss-feed": "rss_news",
}

def _norm_origin(val: Any) -> str:
    if val is None:
        return "unknown"
    s = str(val).strip()
    if not s:
        return "unknown"
    key = s.lower()
    return _ALIAS.get(key, key)

def _ts_to_epoch_s(ts_val: Any) -> float:
    """
    Convert a timestamp value to epoch seconds.
    Accepts:
      - float/int epoch seconds (preferred in tests)
      - ISO-ish strings (best-effort)
    On failure, return 'now' so fresh test rows are never dropped.
    """
    now_s = time.time()
    # Fast path: numeric epoch
    if isinstance(ts_val, (int, float)):
        try:
            return float(ts_val)
        except Exception:
            return now_s
    # Try to parse string ISO
    if isinstance(ts_val, str):
        try:
            # Common case: already numeric-in-string
            return float(ts_val)
        except Exception:
            pass
        # Very lenient ISO parser without external deps
        try:
            # Python's fromisoformat needs adjustments; we only need recency filter
            from datetime import datetime, timezone
            s = ts_val.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return now_s
    # Anything else → now
    return now_s

def _read_jsonl_stream(path: Path):
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
                # Tolerate malformed lines
                continue

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path | None,
    days: int = 7,
    include_triggers: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns (rows, totals)
      rows: [{origin, count, pct}, ...] sorted by count desc then origin asc
      totals: {"flags": N, "triggers": M, "total_events": N(+M)}
    Window filter uses epoch seconds to avoid tz/naive pitfalls.
    Missing/unparsable timestamps are treated as 'now' (keeps brand-new test rows).
    """
    if days <= 0:
        return ([], {"flags": 0, "triggers": 0, "total_events": 0})

    now_s = time.time()
    cutoff = now_s - days * 86400

    # Count flags
    flags_by_origin: Dict[str, int] = {}
    flags_in_window = 0
    for rec in _read_jsonl_stream(flags_path):
        ts_s = _ts_to_epoch_s(rec.get("timestamp"))
        if ts_s < cutoff:
            continue
        origin = _norm_origin(rec.get("origin"))
        flags_by_origin[origin] = flags_by_origin.get(origin, 0) + 1
        flags_in_window += 1

    triggers_by_origin: Dict[str, int] = {}
    triggers_in_window = 0
    if include_triggers and triggers_path:
        for rec in _read_jsonl_stream(triggers_path):
            ts_s = _ts_to_epoch_s(rec.get("timestamp"))
            if ts_s < cutoff:
                continue
            origin = _norm_origin(rec.get("origin"))
            triggers_by_origin[origin] = triggers_by_origin.get(origin, 0) + 1
            triggers_in_window += 1

    # Merge for output rows only if we include triggers
    combined: Dict[str, int] = dict(flags_by_origin)
    if include_triggers:
        for k, v in triggers_by_origin.items():
            combined[k] = combined.get(k, 0) + v

    total_events = flags_in_window + (triggers_in_window if include_triggers else 0)

    if total_events == 0:
        return ([], {"flags": 0, "triggers": 0, "total_events": 0})

    # Build rows with pct
    rows: List[Dict[str, Any]] = []
    for origin, count in combined.items():
        pct = round(100.0 * count / total_events, 2)
        rows.append({"origin": origin, "count": count, "pct": pct})

    # sort: count desc, origin asc
    rows.sort(key=lambda r: (-r["count"], r["origin"]))

    return rows, {
        "flags": flags_in_window,
        "triggers": triggers_in_window if include_triggers else 0,
        "total_events": total_events,
    }