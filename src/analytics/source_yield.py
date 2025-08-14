#!/usr/bin/env python3
"""
Source Yield Scoring & Rate-Limit Budget Planner

Computes per-origin yield scores and suggests API budget allocation
based on flags and triggers in the given time window.

Formula:
    trigger_rate_o = triggers_o / max(flags_o, 1)
    volume_share_o = flags_o / max(total_flags, 1)
    yield_score_o  = alpha * trigger_rate_o + (1 - alpha) * volume_share_o

Budget plan:
    - Only origins with flags >= min_events are eligible
    - Normalize yield scores to percentages
    - Sort desc by pct
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from .origin_utils import extract_origin, parse_ts


def _load_jsonl(path: Path):
    """Tolerant JSONL loader."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def compute_source_yield(flags_path: Path, triggers_path: Path,
                         days: int = 7, min_events: int = 5, alpha: float = 0.7) -> dict:
    """
    Compute source yield scores and budget plan.

    Args:
        flags_path: Path to retraining_log.jsonl
        triggers_path: Path to retraining_triggered.jsonl
        days: Lookback window in days
        min_events: Min flags required to be eligible
        alpha: Blend between trigger_rate and volume_share

    Returns:
        dict ready for JSON response.
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=days)

    flags_log = _load_jsonl(flags_path)
    triggers_log = _load_jsonl(triggers_path)

    flags_count = defaultdict(int)
    triggers_count = defaultdict(int)

    # Process flags
    for row in flags_log:
        ts = parse_ts(row.get("timestamp"))
        if ts is None:
            ts = now
        if ts < window_start:
            continue
        origin = extract_origin(row) or "unknown"
        flags_count[origin] += 1

    # Process triggers
    for row in triggers_log:
        ts = parse_ts(row.get("timestamp"))
        if ts is None:
            ts = now
        if ts < window_start:
            continue
        origin = extract_origin(row) or "unknown"
        triggers_count[origin] += 1

    total_flags = sum(flags_count.values())
    total_triggers = sum(triggers_count.values())

    origins_data = []
    for origin in sorted(flags_count.keys() | triggers_count.keys()):
        f = flags_count.get(origin, 0)
        t = triggers_count.get(origin, 0)
        trigger_rate = t / max(f, 1)
        volume_share = f / max(total_flags, 1)
        yield_score = alpha * trigger_rate + (1 - alpha) * volume_share
        eligible = f >= min_events
        origins_data.append({
            "origin": origin,
            "flags": f,
            "triggers": t,
            "trigger_rate": round(trigger_rate, 6),
            "yield_score": round(yield_score, 6),
            "eligible": eligible
        })

    # Budget plan: only eligible origins
    eligible_origins = [o for o in origins_data if o["eligible"]]
    total_yield = sum(o["yield_score"] for o in eligible_origins)
    budget_plan = []
    if total_yield > 0:
        for o in sorted(eligible_origins, key=lambda x: x["yield_score"], reverse=True):
            pct = (o["yield_score"] / total_yield) * 100
            budget_plan.append({
                "origin": o["origin"],
                "pct": round(pct, 1)
            })

    return {
        "window_days": days,
        "totals": {
            "flags": total_flags,
            "triggers": total_triggers
        },
        "origins": sorted(origins_data, key=lambda x: x["yield_score"], reverse=True),
        "budget_plan": budget_plan,
        "notes": [
            f"Origins with < {min_events} events are excluded from budget_plan.",
            "yield_score blends conversion (trigger_rate) and volume."
        ]
    }
