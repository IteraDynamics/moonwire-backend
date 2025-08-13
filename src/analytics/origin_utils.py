# src/analytics/origin_utils.py

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, Tuple, Any, Optional

# --- origin normalization ---
_ALIAS = {
    "twitter_api": "twitter",
    "Twitter": "twitter",
    "TWITTER": "twitter",
    "rss": "rss_news",
    "RSS": "rss_news",
    "reddit_api": "reddit",
    "Reddit": "reddit",
}

def _norm_origin(s: Optional[str]) -> str:
    if not s:
        return "unknown"
    s = str(s).strip()
    if not s:
        return "unknown"
    return _ALIAS.get(s, s.lower())

# --- timestamp parsing ---
def _parse_ts(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    # epoch seconds (int/float or numeric string)
    try:
        ts = float(val)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    # ISO 8601 (with optional trailing Z)
    try:
        s = str(val)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

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
                # Skip malformed lines
                continue

def _within_window(ts: Optional[datetime], cutoff: datetime) -> bool:
    return bool(ts and ts >= cutoff)

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int = 7,
    include_triggers: bool = True,
) -> Tuple[list, Dict[str, int]]:
    """
    Aggregate per-origin counts over the given window.

    Returns:
      rows: list of {origin, count, pct}
      totals: {"flags": int, "triggers": int, "total_events": int}
    """
    if days <= 0:
        raise ValueError("days must be >= 1")

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    counts: Counter[str] = Counter()
    flags_count = 0
    triggers_count = 0

    # ---- flags (always counted) ----
    for rec in _iter_jsonl(flags_path):
        ts = _parse_ts(rec.get("timestamp"))
        if not _within_window(ts, cutoff):
            continue
        origin = _norm_origin(rec.get("origin"))
        counts[origin] += 1
        flags_count += 1

    # ---- triggers (optional merge) ----
    if include_triggers:
        for rec in _iter_jsonl(triggers_path):
            ts = _parse_ts(rec.get("timestamp"))
            if not _within_window(ts, cutoff):
                continue
            origin = _norm_origin(rec.get("origin"))
            counts[origin] += 1
            triggers_count += 1

    total_events = flags_count + (triggers_count if include_triggers else 0)

    # Convert to sorted rows
    rows = []
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
        "triggers": (triggers_count if include_triggers else 0),
        "total_events": total_events,
    }
    return rows, totals