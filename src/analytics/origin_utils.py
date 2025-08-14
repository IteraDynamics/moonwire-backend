from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List
from datetime import datetime, timedelta, timezone
import json

# --- Origin alias map ---
_ALIAS = {
    "twitter_api": "twitter",
    "twitter": "twitter",
    "Twitter": "twitter",
    "rss": "rss_news",
    "rss_news": "rss_news",
    "reddit": "reddit",
    "Reddit": "reddit",
}

def _norm_origin(raw: Any) -> str:
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    if not s:
        return "unknown"
    return _ALIAS.get(s, s.lower())

def tolerant_origin(row: Dict[str, Any]) -> str:
    """
    Extract origin from a row with schema tolerance.
    Tries keys: origin, source, signal_origin, meta.origin, metadata.source
    Defaults to 'unknown' if none found.
    """
    for key in ("origin", "source", "signal_origin"):
        if key in row and row[key]:
            return _norm_origin(row[key])

    # nested dicts
    for nested_key in ("meta", "metadata"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            for sub_key in ("origin", "source"):
                if sub_key in nested and nested[sub_key]:
                    return _norm_origin(nested[sub_key])

    return "unknown"

def _parse_ts(val: Any) -> datetime | None:
    """
    Accept:
      - float/int epoch seconds
      - numeric strings (e.g., "1723558387.598")
      - ISO-8601 strings (with or without Z)
    Return timezone-aware UTC datetime or None on failure.
    """
    if val is None:
        return None

    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        except Exception:
            return None

    try:
        s = str(val).strip()
        # accept numeric strings as epoch seconds
        if s.replace(".", "", 1).isdigit():
            try:
                return datetime.fromtimestamp(float(s), tz=timezone.utc)
            except Exception:
                pass

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

def tolerant_ts(row: Dict[str, Any]) -> datetime | None:
    """
    Extract timestamp from row with tolerance for schema variations.
    Looks for: timestamp, ts, created_at, meta.timestamp, metadata.created_at
    """
    for key in ("timestamp", "ts", "created_at"):
        if key in row and row[key]:
            return _parse_ts(row[key])

    for nested_key in ("meta", "metadata"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            for sub_key in ("timestamp", "ts", "created_at"):
                if sub_key in nested and nested[sub_key]:
                    return _parse_ts(nested[sub_key])

    return None

def _within_window(ts: datetime | None, now_utc: datetime, days: int) -> bool:
    if ts is None:
        return False
    return ts >= (now_utc - timedelta(days=days))

def _stream_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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
                # tolerate malformed lines
                continue

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int,
    include_triggers: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns (rows, totals)
      rows: [{"origin": str, "count": int, "pct": float}, ...] sorted by count desc then origin asc
      totals: {"flags": int, "triggers": int, "total_events": int}
    """
    now_utc = datetime.now(timezone.utc)

    # Count flags
    flag_counts: Dict[str, int] = {}
    n_flags = 0
    for row in _stream_jsonl(flags_path):
        ts = _parse_ts(row.get("timestamp"))
        if not _within_window(ts, now_utc, days):
            continue
        org = _norm_origin(row.get("origin"))
        flag_counts[org] = flag_counts.get(org, 0) + 1
        n_flags += 1

    # Count triggers (optional)
    trig_counts: Dict[str, int] = {}
    n_trig = 0
    if include_triggers:
        for row in _stream_jsonl(triggers_path):
            ts = _parse_ts(row.get("timestamp"))
            if not _within_window(ts, now_utc, days):
                continue
            org = _norm_origin(row.get("origin"))
            trig_counts[org] = trig_counts.get(org, 0) + 1
            n_trig += 1

    # Merge counts per origin
    combined: Dict[str, int] = dict(flag_counts)
    for org, c in trig_counts.items():
        combined[org] = combined.get(org, 0) + c

    total_events = n_flags + (n_trig if include_triggers else 0)

    # Build rows with percentages
    rows: List[Dict[str, Any]] = []
    if total_events > 0:
        for org, c in combined.items():
            pct = round(100.0 * c / total_events, 2)
            rows.append({"origin": org, "count": c, "pct": pct})

        # sort: count desc, origin asc
        rows.sort(key=lambda r: (-r["count"], r["origin"]))
    else:
        rows = []

    totals = {
        "flags": n_flags,
        "triggers": (n_trig if include_triggers else 0),
        "total_events": total_events,
    }
    return rows, totals