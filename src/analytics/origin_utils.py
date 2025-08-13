# src/analytics/origin_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timezone, timedelta


# ---- helpers ---------------------------------------------------------------

def _parse_ts(raw) -> datetime | None:
    """
    Accepts epoch (int/float) or ISO string. Returns UTC datetime, or None if unparsable.
    """
    if raw is None:
        return None
    # epoch?
    try:
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    except Exception:
        pass
    # ISO?
    try:
        s = str(raw)
        if s.endswith("Z"):
            # datetime.fromisoformat doesn't accept 'Z'
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


_ALIAS_MAP = {
    "twitter_api": "twitter",
    "twitter": "twitter",
    "Twitter": "twitter",
    "rss": "rss_news",
    "rss_news": "rss_news",
}

def normalize_origin(val: str | None) -> str:
    if not val:
        return "unknown"
    v = str(val).strip()
    if not v:
        return "unknown"
    return _ALIAS_MAP.get(v, _ALIAS_MAP.get(v.lower(), v.lower()))


# ---- core -------------------------------------------------------------------

def _stream_jsonl(path: Path):
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
                # tolerate bad lines
                continue


def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int = 7,
    include_triggers: bool = True,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Reads flags (retraining_log.jsonl) and optionally triggers (retraining_triggered.jsonl),
    filters to last N days, aggregates per origin, and returns:

      rows: [{ "origin": str, "count": int, "pct": float }, ...]  (sorted)
      totals: { "flags": int, "triggers": int, "total_events": int }
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    counts: Dict[str, int] = {}

    # Count flags
    flags_count = 0
    for rec in _stream_jsonl(flags_path):
        ts = _parse_ts(rec.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        origin = normalize_origin(rec.get("origin"))
        counts[origin] = counts.get(origin, 0) + 1
        flags_count += 1

    # Count triggers (optional)
    triggers_count = 0
    if include_triggers:
        for rec in _stream_jsonl(triggers_path):
            ts = _parse_ts(rec.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            origin = normalize_origin(rec.get("origin"))
            counts[origin] = counts.get(origin, 0) + 1
            triggers_count += 1

    total_events = flags_count + (triggers_count if include_triggers else 0)

    # Build rows w/ percentages
    rows: List[Dict] = []
    if total_events > 0:
        for origin, cnt in counts.items():
            pct = round(100.0 * cnt / total_events, 2)
            rows.append({"origin": origin, "count": cnt, "pct": pct})

        # sort: count desc, then origin asc
        rows.sort(key=lambda r: (-r["count"], r["origin"]))
    else:
        rows = []

    totals = {
        "flags": flags_count,
        "triggers": triggers_count if include_triggers else 0,
        "total_events": total_events,
    }
    return rows, totals