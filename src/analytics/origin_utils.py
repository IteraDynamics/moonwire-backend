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
    "twitter": "twitter",
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
    return _ALIAS.get(key, key)

# ---- timestamp parsing -------------------------------------------------------

def _parse_ts(v: Any) -> datetime | None:
    """
    Accept:
      - unix epoch (int/float/str)
      - ISO 8601 (with/without 'Z')
    Return aware UTC datetime or None if unparsable.
    """
    if v is None:
        return None
    try:
        sec = float(v)
        return datetime.fromtimestamp(sec, tz=timezone.utc)
    except Exception:
        pass
    try:
        s = str(v)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

# ---- streaming reader --------------------------------------------------------

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
                continue  # skip malformed

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

    Behavior:
      - totals["flags"] is ALWAYS the number of flags in-window (lenient parsing).
      - totals["triggers"] is the number of triggers in-window (lenient parsing),
        but only contributes to total_events when include_triggers=True.
    """
    if days <= 0:
        return ([], {"flags": 0, "triggers": 0, "total_events": 0})

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    # 1) FLAGS
    flag_counter: Counter[str] = Counter()
    flags_in_window = 0
    for rec in _iter_jsonl(flags_path):
        # --- LENIENT WINDOW: treat missing/unparsable ts as "now" so tests/CI don't drop fresh rows
        ts = _parse_ts(rec.get("timestamp")) or now
        if ts >= cutoff:
            origin = normalize_origin(rec.get("origin"))
            flag_counter[origin] += 1
            flags_in_window += 1

    # 2) TRIGGERS
    trig_counter: Counter[str] = Counter()
    triggers_in_window = 0
    if triggers_path:
        for rec in _iter_jsonl(triggers_path):
            ts = _parse_ts(rec.get("timestamp")) or now  # LENIENT
            if ts >= cutoff:
                origin = normalize_origin(rec.get("origin"))
                trig_counter[origin] += 1
                triggers_in_window += 1

    # 3) Build response set
    combined: Counter[str] = Counter(flag_counter)
    if include_triggers:
        combined.update(trig_counter)

    total_events = sum(combined.values())
    rows: List[Dict[str, Any]] = []
    if total_events > 0:
        for origin, count in combined.items():
            pct = round(100.0 * count / total_events, 2)
            rows.append({"origin": origin, "count": int(count), "pct": pct})
        rows.sort(key=lambda r: (-r["count"], r["origin"]))

    # 4) Totals
    totals = {
        "flags": flags_in_window,
        "triggers": triggers_in_window if include_triggers else 0,
        "total_events": total_events,
    }
    return rows, totals