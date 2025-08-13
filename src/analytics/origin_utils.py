# src/analytics/origin_utils.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, Tuple, List


# ----- Origin normalization -----
_ALIASES = {
    "twitter_api": "twitter",
    "Twitter": "twitter",
    "rss": "rss_news",
    "news": "rss_news",
}
def _norm(origin: str | None) -> str:
    if not origin:
        return "unknown"
    o = str(origin).strip()
    return _ALIASES.get(o, o.lower())


# ----- JSONL helpers -----
def _iter_jsonl(path: Path) -> Iterable[dict]:
    """Yield parsed objects from a JSONL file, skipping malformed lines."""
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


def _to_epoch(ts_val: Any) -> float | None:
    """Accept float epoch or ISO8601 (with optional Z); return epoch seconds or None."""
    # float/epoch
    try:
        return float(ts_val)
    except Exception:
        pass
    # ISO string
    try:
        s = str(ts_val)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _count_by_origin(path: Path, cutoff_epoch: float) -> Tuple[Dict[str, int], int]:
    counts: Dict[str, int] = {}
    total = 0
    for row in _iter_jsonl(path):
        ts = _to_epoch(row.get("timestamp"))
        if ts is None or ts < cutoff_epoch:
            continue
        origin = _norm(row.get("origin"))
        counts[origin] = counts.get(origin, 0) + 1
        total += 1
    return counts, total


# ----- Public API -----
def compute_origin_breakdown(
    flags_path: Path,
    triggers_path: Path,
    *,
    days: int,
    include_triggers: bool,
) -> tuple[List[dict], dict]:
    """
    Return (rows, totals) for origin analytics.

    rows: [{"origin": str, "count": int, "pct": float}], sorted by count desc then origin asc
    totals: {"flags": int, "triggers": int, "total_events": int}
    """
    if days < 1:
        raise ValueError("days must be >= 1")

    now = datetime.now(timezone.utc).timestamp()
    cutoff = now - days * 86400

    # Count each stream independently (prevents double counting)
    flag_counts, flags_total = _count_by_origin(flags_path, cutoff)
    trig_counts, triggers_total = _count_by_origin(triggers_path, cutoff)

    total_events = flags_total + (triggers_total if include_triggers else 0)

    # Merge for display only (flags + optional triggers)
    combined: Dict[str, int] = dict(flag_counts)
    if include_triggers:
        for k, v in trig_counts.items():
            combined[k] = combined.get(k, 0) + v

    rows: List[dict] = []
    for origin, cnt in combined.items():
        pct = 0.0 if total_events == 0 else round(100.0 * cnt / total_events, 2)
        rows.append({"origin": origin, "count": cnt, "pct": pct})

    rows.sort(key=lambda r: (-r["count"], r["origin"]))

    totals = {
        "flags": flags_total,
        "triggers": triggers_total,
        "total_events": total_events,
    }
    return rows, totals