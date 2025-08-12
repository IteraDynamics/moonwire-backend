# src/analytics/origin_utils.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List, Any
from datetime import datetime, timedelta, timezone


# ---- timestamp helpers --------------------------------------------------------

def _parse_ts(val) -> datetime | None:
    """Accept float/epoch or ISO(ish). Returns aware UTC datetime, or None."""
    if val is None:
        return None
    # epoch number?
    try:
        t = float(val)
        return datetime.fromtimestamp(t, tz=timezone.utc)
    except Exception:
        pass
    # ISO string?
    try:
        s = str(val)
        # tolerate trailing 'Z'
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


# ---- origin normalization -----------------------------------------------------

_ALIAS = {
    "twitter_api": "twitter",
    "tweet": "twitter",
    "rss": "rss_news",
    "rss-news": "rss_news",
    "news_rss": "rss_news",
    "reddit_api": "reddit",
}

def _normalize_origin(raw: Any) -> str:
    if not raw:
        return "unknown"
    s = str(raw).strip().lower()
    return _ALIAS.get(s, s) or "unknown"


# ---- core computation ---------------------------------------------------------

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int,
    include_triggers: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Stream JSONL logs and compute per-origin counts within the lookback window.

    Returns:
      rows:  [{origin, count, pct}, ...]  (sorted count desc, origin asc)
      totals: {"flags": int, "triggers": int, "total_events": int}
    """
    if days <= 0:
        raise ValueError("days must be >= 1")

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    counts: Dict[str, int] = {}
    flags_count = 0
    triggers_count = 0

    # ---- flags (always included) ----
    if flags_path.exists():
        with flags_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                ts = _parse_ts(rec.get("timestamp"))
                if not ts or ts < cutoff:
                    continue

                origin = _normalize_origin(rec.get("origin"))
                counts[origin] = counts.get(origin, 0) + 1
                flags_count += 1

    # ---- triggers (optional) ----
    if include_triggers and triggers_path.exists():
        with triggers_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                ts = _parse_ts(rec.get("timestamp"))
                if not ts or ts < cutoff:
                    continue

                origin = _normalize_origin(rec.get("origin"))
                counts[origin] = counts.get(origin, 0) + 1
                triggers_count += 1

    total_events = flags_count + (triggers_count if include_triggers else 0)

    # Build rows with pct; apply 2dp rounding
    rows: List[Dict[str, Any]] = []
    if total_events > 0:
        for origin, cnt in counts.items():
            pct = round(100.0 * cnt / total_events, 2)
            rows.append({"origin": origin, "count": cnt, "pct": pct})
        # sort: count desc, origin asc
        rows.sort(key=lambda r: (-r["count"], r["origin"]))
    else:
        # No events → empty origins list
        rows = []

    totals = {
        "flags": flags_count,
        "triggers": (triggers_count if include_triggers else 0),
        "total_events": total_events,
    }
    return rows, totals