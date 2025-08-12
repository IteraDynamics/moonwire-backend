from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Tuple, Optional

# --- Normalization map: add aliases here as needed ---
_ORIGIN_ALIASES = {
    "twitter_api": "twitter",
    "twitterapi": "twitter",
    "twitter": "twitter",
    "reddit": "reddit",
    "rss": "rss_news",
    "rss_news": "rss_news",
    "market": "market_feed",
    "market_feed": "market_feed",
}


def _norm_origin(raw: object) -> str:
    if raw is None:
        return "unknown"
    s = str(raw).strip().lower()
    if not s:
        return "unknown"
    return _ORIGIN_ALIASES.get(s, s)


def _parse_ts(val: object) -> Optional[datetime]:
    """
    Accepts:
      - float / int epoch seconds
      - numeric strings of epoch seconds
      - ISO-8601 strings, with or without 'Z'
    Returns timezone-aware UTC datetime or None on failure.
    """
    if val is None:
        return None

    # epoch numbers or numeric strings
    try:
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        sval = str(val).strip()
        if sval.replace(".", "", 1).isdigit():  # "1691060400" or "1691060400.123"
            return datetime.fromtimestamp(float(sval), tz=timezone.utc)
    except Exception:
        pass

    # ISO strings
    try:
        s = str(val).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
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


def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Optional[Path],
    days: int,
    include_triggers: bool,
) -> Tuple[list, Dict[str, int]]:
    """
    Returns:
      origins: list of {"origin": str, "count": int, "pct": float}
      totals:  {"total_events": int, "flags": int, "triggers": int}
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    counts: Dict[str, int] = {}
    flags_count = 0
    triggers_count = 0

    # ---- Process flags ----
    for rec in _iter_jsonl(flags_path):
        ts = _parse_ts(rec.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        origin = _norm_origin(rec.get("origin"))
        counts[origin] = counts.get(origin, 0) + 1
        flags_count += 1

    # ---- Process triggers if requested and path provided ----
    if include_triggers and triggers_path is not None:
        for rec in _iter_jsonl(triggers_path):
            ts = _parse_ts(rec.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            origin = _norm_origin(rec.get("origin"))
            counts[origin] = counts.get(origin, 0) + 1
            triggers_count += 1

    total_events = sum(counts.values())

    # Build sorted list
    origins_list = []
    for origin, cnt in counts.items():
        pct = round(100.0 * cnt / total_events, 2) if total_events > 0 else 0.0
        origins_list.append({"origin": origin, "count": cnt, "pct": pct})

    # Sort by count desc, then origin asc for stability
    origins_list.sort(key=lambda x: (-x["count"], x["origin"]))

    totals = {
        "total_events": total_events,
        "flags": flags_count,
        "triggers": triggers_count,
    }
    return origins_list, totals