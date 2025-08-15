from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any

from src.analytics.origin_utils import (
    stream_jsonl,
    extract_origin,
    parse_ts
)


def compute_source_yield(
    flags_path: Path,
    triggers_path: Path,
    days: int = 7,
    min_events: int = 5,
    alpha: float = 0.7
) -> Dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    flags_by_origin = defaultdict(int)
    triggers_by_origin = defaultdict(int)

    # Parse flags
    for record in stream_jsonl(flags_path):
        ts = parse_ts(record.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        origin = extract_origin(record.get("origin") or record.get("source") or record.get("meta", {}).get("origin") or record.get("metadata", {}).get("source"))
        flags_by_origin[origin] += 1

    total_flags = sum(flags_by_origin.values())

    # Parse triggers
    if triggers_path.exists():
        for record in stream_jsonl(triggers_path):
            ts = parse_ts(record.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            origin = extract_origin(record.get("origin") or record.get("source") or record.get("meta", {}).get("origin") or record.get("metadata", {}).get("source"))
            triggers_by_origin[origin] += 1

    # Build per-origin stats
    origins = []
    for origin, flags in flags_by_origin.items():
        triggers = triggers_by_origin.get(origin, 0)
        trigger_rate = triggers / max(flags, 1)
        volume_share = flags / max(total_flags, 1)
        yield_score = round(alpha * trigger_rate + (1 - alpha) * volume_share, 6)
        origins.append({
            "origin": origin,
            "flags": flags,
            "triggers": triggers,
            "trigger_rate": round(trigger_rate, 6),
            "yield_score": yield_score,
            "eligible": flags >= min_events
        })

    # Compute budget plan
    eligible = [o for o in origins if o["eligible"]]
    total_yield = sum(o["yield_score"] for o in eligible) or 1
    budget_plan = [
        {
            "origin": o["origin"],
            "pct": round(100 * o["yield_score"] / total_yield, 1)
        }
        for o in sorted(eligible, key=lambda o: o["yield_score"], reverse=True)
    ]

    return {
        "window_days": days,
        "totals": {
            "flags": total_flags,
            "triggers": sum(triggers_by_origin.values())
        },
        "origins": origins,
        "budget_plan": budget_plan,
        "notes": [
            f"Origins with < min_events are excluded from budget_plan.",
            f"yield_score blends conversion (trigger_rate) and volume."
        ]
    }