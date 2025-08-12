from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, Tuple, List, Any, Optional

# Robust timestamp parsing: supports epoch seconds or ISO (with or w/o Z)
def _parse_ts(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    # epoch seconds
    try:
        ts = float(val)
        if 0 < ts < 10_000_000_000:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    # ISO
    try:
        s = str(val)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

# --- origin normalization ---
_ORIGIN_ALIASES = {
    "twitter_api": "twitter",
    "twitterapi": "twitter",
    "tweet": "twitter",
    "rss": "rss_news",
    "news": "rss_news",
    "reddit_api": "reddit",
    "markets": "market_feed",
    "market": "market_feed",
}
def normalize_origin(raw: Any) -> str:
    if not raw:
        return "unknown"
    s = str(raw).strip().lower()
    return _ORIGIN_ALIASES.get(s, s)

def _stream_jsonl(path: Path) -> Iterable[dict]:
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
                continue

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int = 7,
    include_triggers: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns (origins, totals)

    origins: [{ "origin": str, "count": int, "pct": float }, ...]  — counts are
             flags+triggers if include_triggers=True, otherwise flags only.
    totals:  { "flags": int, "triggers": int, "total_events": int }
    """
    if days <= 0:
        return [], {"flags": 0, "triggers": 0, "total_events": 0}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    # Count flags and triggers separately
    per_origin_flags: Dict[str, int] = {}
    per_origin_triggers: Dict[str, int] = {}
    flags_total = 0
    triggers_total = 0

    # Flags
    for rec in _stream_jsonl(flags_path):
        ts = _parse_ts(rec.get("timestamp"))
        if not ts or ts < cutoff:
            continue
        origin = normalize_origin(rec.get("origin"))
        per_origin_flags[origin] = per_origin_flags.get(origin, 0) + 1
        flags_total += 1

    # Triggers
    if include_triggers:
        for rec in _stream_jsonl(triggers_path):
            ts = _parse_ts(rec.get("timestamp"))
            if not ts or ts < cutoff:
                continue
            origin = normalize_origin(rec.get("origin"))
            per_origin_triggers[origin] = per_origin_triggers.get(origin, 0) + 1
            triggers_total += 1

    # Merge per-origin counts for output list (respect include_triggers)
    all_origins = set(per_origin_flags) | (set(per_origin_triggers) if include_triggers else set())
    combined: List[Dict[str, Any]] = []

    total_events = flags_total + (triggers_total if include_triggers else 0)
    if total_events == 0:
        return [], {"flags": flags_total, "triggers": triggers_total if include_triggers else 0, "total_events": 0}

    for o in all_origins:
        c = per_origin_flags.get(o, 0) + (per_origin_triggers.get(o, 0) if include_triggers else 0)
        pct = round(100.0 * c / total_events, 2) if total_events else 0.0
        combined.append({"origin": o, "count": c, "pct": pct})

    # Sort by count desc, then origin asc
    combined.sort(key=lambda x: (-x["count"], x["origin"]))

    return combined, {"flags": flags_total, "triggers": triggers_total if include_triggers else 0, "total_events": total_events}