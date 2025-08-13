# src/analytics/origin_utils.py
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

# ---- origin normalization ----------------------------------------------------

_ALIAS = {
    "twitter_api": "twitter",
    "Twitter": "twitter",
    "TWITTER": "twitter",
    "rss": "rss_news",
    "RSS": "rss_news",
}

def normalize_origin(val: Any) -> str:
    if val is None:
        return "unknown"
    s = str(val).strip()
    if not s:
        return "unknown"
    key = s.lower()
    # map aliases (case-insensitive)
    if key in _ALIAS:
        return _ALIAS[key]
    # also map canonical lowercase of alias values
    return key

# ---- timestamp parsing -------------------------------------------------------

def _parse_ts(v: Any) -> datetime | None:
    """
    Accepts:
      - unix epoch (int/float/str)
      - ISO 8601 string (with or without 'Z')
    Returns UTC datetime or None if unparsable.
    """
    if v is None:
        return None
    # unix epoch (fast path)
    try:
        sec = float(v)
        return datetime.fromtimestamp(sec, tz=timezone.utc)
    except Exception:
        pass
    # ISO-ish
    try:
        s = str(v)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

# ---- streaming readers -------------------------------------------------------

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
                # skip malformed
                continue

# ---- core computation --------------------------------------------------------

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path | None,
    *,
    days: int,
    include_triggers: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns (rows, totals):
      rows: [{origin, count, pct}]
      totals: {"flags": int, "triggers": int, "total_events": int}

    IMPORTANT:
      - `totals["flags"]` is ALWAYS the number of flag events in-window,
        even when include_triggers=False.
      - `totals["triggers"]` is the number of trigger events in-window,
        but only added into total_events when include_triggers=True.
    """
    if days <= 0:
        # let router raise 400, but keep this defensive
        return ([], {"flags": 0, "triggers": 0, "total_events": 0})

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    # 1) Collect and count FLAGS (always)
    flag_counter: Counter[str] = Counter()
    flags_in_window = 0
    for rec in _iter_jsonl(flags_path):
        ts = _parse_ts(rec.get("timestamp"))
        if not ts or ts < cutoff:
            continue
        origin = normalize_origin(rec.get("origin"))
        flag_counter[origin] += 1
        flags_in_window += 1

    # 2) Collect and count TRIGGERS (in-window), but their inclusion in totals depends on include_triggers
    trig_counter: Counter[str] = Counter()
    triggers_in_window = 0
    if triggers_path:
        for rec in _iter_jsonl(triggers_path):
            ts = _parse_ts(rec.get("timestamp"))
            if not ts or ts < cutoff:
                continue
            origin = normalize_origin(rec.get("origin"))
            trig_counter[origin] += 1
            triggers_in_window += 1

    # 3) Build the rows we include for the response
    # If include_triggers=True → origins = flags + triggers; else origins = flags only.
    combined: Counter[str] = Counter(flag_counter)
    if include_triggers:
        combined.update(trig_counter)

    total_events = sum(combined.values())  # what the chart & pct are based on
    rows: List[Dict[str, Any]] = []
    if total_events > 0:
        for origin, count in combined.items():
            pct = round(100.0 * count / total_events, 2)
            rows.append({"origin": origin, "count": int(count), "pct": pct})
        # sort by count desc, origin asc
        rows.sort(key=lambda r: (-r["count"], r["origin"]))
    else:
        rows = []

    # 4) Totals:
    # - flags = flags_in_window (ALWAYS the count of flags found)
    # - triggers = triggers_in_window (informational)
    # - total_events = len(origins included in this response)
    totals = {
        "flags": flags_in_window,
        "triggers": triggers_in_window if include_triggers else 0 if triggers_in_window >= 0 else 0,
        "total_events": total_events,
    }

    return rows, totals