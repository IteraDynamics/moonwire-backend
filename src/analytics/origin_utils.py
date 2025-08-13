# src/analytics/origin_utils.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Iterable, Any
from datetime import datetime, timezone

# --------- parsing helpers ---------
def _parse_ts(val: Any) -> float | None:
    """Return UTC epoch seconds or None."""
    if val is None:
        return None
    # numeric epoch
    try:
        return float(val)
    except Exception:
        pass
    # ISO string (accepts Z)
    try:
        s = str(val)
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


def _iter_jsonl(path: Path) -> Iterable[dict]:
    """Yield JSON objects line-by-line, tolerant of bad lines."""
    if not path or not Path(path).exists():
        return
    with Path(path).open("r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                # skip malformed
                continue


def _normalize_origin(raw: Any) -> str:
    if not raw:
        return "unknown"
    s = str(raw).strip().lower()
    # alias map
    if s in ("twitter_api", "twitterapi", "tw", "twt", "twitter"):
        return "twitter"
    if s in ("rss", "rss-news", "news_rss", "rssnews", "feed", "rss_feed"):
        return "rss_news"
    if s in ("reddit_api", "redditapi", "rdt", "reddit"):
        return "reddit"
    if s in ("market", "marketfeed", "market-feed", "market_data", "marketdata"):
        return "market_feed"
    return s


# --------- core computation ---------
def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int,
    include_triggers: bool,
) -> Tuple[list[dict], dict]:
    """
    Stream logs and compute per-origin counts and totals.

    Returns:
      rows: [{"origin": str, "count": int, "pct": float}, ...]  (sorted count desc, origin asc)
      totals: {"total_events": int, "flags": int, "triggers": int}
    """
    if days <= 0:
        # the router validates ge=1, but be defensive
        return [], {"total_events": 0, "flags": 0, "triggers": 0}

    now = datetime.now(tz=timezone.utc).timestamp()
    cutoff = now - days * 86400.0

    counts: Dict[str, int] = {}
    flags_total = 0
    triggers_total = 0

    # ----- flags (ALWAYS included) -----
    for rec in _iter_jsonl(flags_path):
        ts = _parse_ts(rec.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        origin = _normalize_origin(rec.get("origin"))
        counts[origin] = counts.get(origin, 0) + 1
        flags_total += 1

    # ----- triggers (OPTIONAL, added on top of flags) -----
    if include_triggers:
        for rec in _iter_jsonl(triggers_path):
            ts = _parse_ts(rec.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            origin = _normalize_origin(rec.get("origin"))
            counts[origin] = counts.get(origin, 0) + 1
            triggers_total += 1

    total_events = flags_total + (triggers_total if include_triggers else 0)

    # build rows
    rows = []
    for origin, cnt in counts.items():
        pct = round(100.0 * cnt / total_events, 2) if total_events > 0 else 0.0
        rows.append({"origin": origin, "count": cnt, "pct": pct})

    # sort: count desc, origin asc
    rows.sort(key=lambda r: (-r["count"], r["origin"]))

    totals = {
        "total_events": total_events,
        "flags": flags_total,
        "triggers": triggers_total if include_triggers else 0,
    }
    return rows, totals