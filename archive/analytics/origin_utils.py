# src/analytics/origin_utils.py
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

# ---------------------------------------------------------------------------
# Internal helpers (prefixed with _)
# ---------------------------------------------------------------------------

def _norm_origin(raw: Any) -> str:
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    if not s:
        return "unknown"
    return _ALIAS.get(s, s.lower())

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

# ---------------------------------------------------------------------------
# Public API (stable names used by other modules/tests)
# ---------------------------------------------------------------------------

def normalize_origin(raw: Any) -> str:
    """Public wrapper around the internal normalizer."""
    return _norm_origin(raw)

def extract_origin(raw: Any) -> str:
    """Backward-compatible alias many modules use."""
    return _norm_origin(raw)

def parse_ts(val: Any) -> datetime | None:
    """Public tolerant timestamp parser."""
    return _parse_ts(val)

def stream_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Public tolerant JSONL reader (yields dict rows)."""
    yield from _stream_jsonl(path)

# Back-compat alias if any legacy code imports this name
tolerant_jsonl_stream = stream_jsonl

# ---------------------------------------------------------------------------
# Analytics: origin breakdown
# ---------------------------------------------------------------------------

def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int,
    include_triggers: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns (rows, totals)
      rows: [{"origin": str, "count": int, "pct": float, "percent": float}, ...]
            sorted by count desc then origin asc
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
        org = _norm_origin(
            row.get("origin")
            or row.get("source")
            or (row.get("meta") or {}).get("origin")
            or (row.get("metadata") or {}).get("source")
        )
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
            org = _norm_origin(
                row.get("origin")
                or row.get("source")
                or (row.get("meta") or {}).get("origin")
                or (row.get("metadata") or {}).get("source")
            )
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
            # Provide both keys to satisfy tests ("pct") and summary code ("percent")
            rows.append({"origin": org, "count": c, "pct": pct, "percent": pct})

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

__all__ = [
    "normalize_origin",
    "extract_origin",
    "parse_ts",
    "stream_jsonl",
    "tolerant_jsonl_stream",
    "compute_origin_breakdown",
]