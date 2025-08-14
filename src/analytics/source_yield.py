from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import json

from src.analytics.origin_utils import tolerant_origin, tolerant_ts


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def compute_source_yield(
    flags_path: Path,
    triggers_path: Path,
    days: int,
    min_events: int,
    alpha: float
) -> Dict[str, Any]:
    """
    Compute per-origin yield score and API budget plan.
    """

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    flags = load_jsonl(flags_path)
    triggers = load_jsonl(triggers_path)

    # Filter by date window
    def in_window(rec):
        ts = tolerant_ts(rec.get("timestamp"))
        if ts is None:
            ts = now
        return ts >= cutoff

    flags = [r for r in flags if in_window(r)]
    triggers = [r for r in triggers if in_window(r)]

    total_flags = len(flags)
    total_triggers = len(triggers)

    # Count per origin
    stats = {}
    for r in flags:
        o = tolerant_origin(r)
        if not o:
            continue
        stats.setdefault(o, {"flags": 0, "triggers": 0})
        stats[o]["flags"] += 1

    for r in triggers:
        o = tolerant_origin(r)
        if not o:
            continue
        stats.setdefault(o, {"flags": 0, "triggers": 0})
        stats[o]["triggers"] += 1

    origins_out = []
    for origin, vals in stats.items():
        f = vals["flags"]
        t = vals["triggers"]
        trigger_rate = t / max(f, 1)
        volume_share = f / max(total_flags, 1)
        yield_score = alpha * trigger_rate + (1 - alpha) * volume_share
        eligible = f >= min_events
        origins_out.append({
            "origin": origin,
            "flags": f,
            "triggers": t,
            "trigger_rate": round(trigger_rate, 6),
            "yield_score": round(yield_score, 6),
            "eligible": eligible
        })

    # Budget plan: normalize yield among eligible origins
    eligible_origins = [o for o in origins_out if o["eligible"]]
    total_yield = sum(o["yield_score"] for o in eligible_origins) or 1.0
    budget_plan = [
        {
            "origin": o["origin"],
            "pct": round((o["yield_score"] / total_yield) * 100, 1)
        }
        for o in sorted(eligible_origins, key=lambda x: x["yield_score"], reverse=True)
    ]

    return {
        "window_days": days,
        "totals": {
            "flags": total_flags,
            "triggers": total_triggers
        },
        "origins": sorted(origins_out, key=lambda x: x["yield_score"], reverse=True),
        "budget_plan": budget_plan,
        "notes": [
            f"Origins with < min_events ({min_events}) are excluded from budget_plan.",
            "yield_score blends conversion (trigger_rate) and volume."
        ]
    }
