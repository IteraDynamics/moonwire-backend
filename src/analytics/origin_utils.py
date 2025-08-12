# src/analytics/origin_utils.py
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional, List
from datetime import datetime, timedelta, timezone

# ---- normalization ----

_ALIAS_MAP = {
    "twitter_api": "twitter",
    "x": "twitter",
    "rss": "rss_news",
    "news": "rss_news",
    "market": "market_feed",
}

def normalize_origin(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    s = str(raw).strip().lower()
    return _ALIAS_MAP.get(s, s)

# ---- parsing helpers ----

def _parse_ts(val) -> Optional[datetime]:
    """Accept unix seconds (int/float) or ISO-8601. Return timezone-aware UTC or None."""
    if val is None:
        return None
    # unix seconds?
    try:
        t = float(val)
        return datetime.fromtimestamp(t, tz=timezone.utc)
    except (TypeError, ValueError):
        pass
    # ISO
    try:
        s = str(val)
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

def _iter_jsonl(path: Path) -> Iterable[dict]:
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
                # tolerate corrupt line
                continue

# ---- aggregation core ----

def aggregate_origins(
    flags_path: Path,
    triggers_path: Path,
    days: int,
    include_triggers: bool,
) -> Tuple[Dict[str, int], int, int]:
    """
    Returns: (counts_by_origin, flags_included, triggers_included)
    Window: now - days.
    """
    now = datetime.now(timezone.utc)
    if days <= 0:
        raise ValueError("days must be positive")
    cutoff = now - timedelta(days=days)

    counts: Dict[str, int] = {}
    flags_included = 0
    triggers_included = 0

    # Flags
    for rec in _iter_jsonl(flags_path):
        ts = _parse_ts(rec.get("timestamp"))
        if not ts or ts < cutoff:
            continue
        origin = normalize_origin(rec.get("origin"))
        counts[origin] = counts.get(origin, 0) + 1
        flags_included += 1

    # Triggers
    if include_triggers:
        for rec in _iter_jsonl(triggers_path):
            ts = _parse_ts(rec.get("timestamp"))
            if not ts or ts < cutoff:
                continue
            origin = normalize_origin(rec.get("origin"))
            counts[origin] = counts.get(origin, 0) + 1
            triggers_included += 1

    return counts, flags_included, triggers_included

def build_breakdown(
    counts: Dict[str, int],
    flags_included: int,
    triggers_included: int,
    min_count: int,
    window_days: int,
) -> dict:
    total_events = sum(counts.values())
    # pct denominator is total_events (before min_count filter), per spec.
    denom = max(total_events, 1)  # avoid div-by-zero; values dropped to 0 later

    rows = [
        {
            "origin": origin,
            "count": cnt,
            "pct": round(100.0 * cnt / denom, 2),
        }
        for origin, cnt in counts.items()
        if cnt >= max(1, min_count)
    ]

    # sort by count desc, then origin asc
    rows.sort(key=lambda x: (-x["count"], x["origin"]))

    return {
        "window_days": window_days,
        "total_events": total_events,
        "origins": rows,
        "included": {"flags": flags_included, "triggers": triggers_included},
    }