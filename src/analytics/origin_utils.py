# src/analytics/origin_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any, Optional
from datetime import datetime, timezone

# --- origin normalization ----------------------------------------------------

_ALIASES = {
    "twitter_api": "twitter",
    "Twitter": "twitter",
    "reddit_api": "reddit",
    "rss": "rss_news",
    "news": "rss_news",
}

def normalize_origin(val: Optional[str]) -> str:
    if not val:
        return "unknown"
    s = str(val).strip()
    return _ALIASES.get(s, s.lower())


# --- timestamp parsing -------------------------------------------------------

def _parse_ts(ts_val: Any) -> Optional[float]:
    """Return UTC epoch seconds, or None if unparsable."""
    if ts_val is None:
        return None
    # numeric epoch seconds
    try:
        return float(ts_val)
    except Exception:
        pass
    # ISO 8601
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


# --- streaming readers -------------------------------------------------------

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
                # ignore malformed lines
                continue


# --- main computation --------------------------------------------------------

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int,
    include_triggers: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Compute counts and share per origin over a lookback window.

    Key behavior:
    - Flags and triggers are tallied from their respective files only.
    - Events outside the window are ignored.
    - Deduplication is by (signal_id, origin) within each stream so repeated rows
      for the same logical event don't inflate counts.
    """
    if days <= 0:
        raise ValueError("days must be >= 1")

    now = datetime.now(timezone.utc).timestamp()
    cutoff = now - days * 86400

    # tallies
    per_origin: Dict[str, int] = {}
    flags_count = 0
    triggers_count = 0

    # de-dup sets (ignore timestamp; first event for a (signal_id, origin) wins)
    seen_flags: set[Tuple[str, str]] = set()
    seen_triggers: set[Tuple[str, str]] = set()

    # --- flags (ONLY from flags_path) ---
    for rec in _iter_jsonl(flags_path):
        ts = _parse_ts(rec.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        origin = normalize_origin(rec.get("origin"))
        sid = str(rec.get("signal_id", ""))
        key = (sid, origin)
        if key in seen_flags:
            continue
        seen_flags.add(key)
        per_origin[origin] = per_origin.get(origin, 0) + 1
        flags_count += 1

    # --- triggers (ONLY from triggers_path) ---
    if include_triggers:
        for rec in _iter_jsonl(triggers_path):
            ts = _parse_ts(rec.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            origin = normalize_origin(rec.get("origin"))
            sid = str(rec.get("signal_id", ""))
            key = (sid, origin)
            if key in seen_triggers:
                continue
            seen_triggers.add(key)
            per_origin[origin] = per_origin.get(origin, 0) + 1
            triggers_count += 1

    total_events = flags_count + (triggers_count if include_triggers else 0)

    rows: List[Dict[str, Any]] = []
    for origin, count in per_origin.items():
        pct = 0.0 if total_events == 0 else round(100.0 * count / total_events, 2)
        rows.append({"origin": origin, "count": count, "pct": pct})

    # sort by count desc, then origin asc
    rows.sort(key=lambda r: (-r["count"], r["origin"]))

    totals = {
        "flags": flags_count,
        "triggers": triggers_count if include_triggers else 0,
        "total_events": total_events,
    }
    return rows, totals