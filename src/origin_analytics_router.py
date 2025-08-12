# src/origin_analytics_router.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import List, Dict, Any

from src.paths import (
    RETRAINING_LOG_PATH,
    RETRAINING_TRIGGERED_LOG_PATH,
)
from src.analytics.origin_utils import compute_origin_breakdown

router = APIRouter(prefix="/internal")


def _coerce_bool(val: bool | str | None, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).lower()
    return s in ("1", "true", "yes", "y", "on")


@router.get("/signal-origins")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days (>=1)"),
    min_count: int = Query(1, ge=0, description="Drop origins below this count (after counting)"),
    include_triggers: bool | str = Query(True, description="Include retraining triggers in counts"),
) -> Dict[str, Any]:
    """
    Return counts and percentages of events by origin over a lookback window.
    Flags always counted; triggers included when include_triggers=true.
    """
    include_triggers_bool = _coerce_bool(include_triggers, True)

    # Compute raw breakdown/totals
    origins_list, totals = compute_origin_breakdown(
        flags_path=Path(RETRAINING_LOG_PATH),
        triggers_path=Path(RETRAINING_TRIGGERED_LOG_PATH),
        days=days,
        include_triggers=include_triggers_bool,
    )

    # Apply min_count filter ONLY to the per-origin rows (NOT to totals)
    if min_count > 0:
        filtered = [row for row in origins_list if row["count"] >= min_count]
    else:
        filtered = origins_list

    # Sort is already handled in compute_origin_breakdown, but re-assert order after filter
    filtered.sort(key=lambda x: (-x["count"], x["origin"]))

    # Build response
    response = {
        "window_days": days,
        "total_events": totals.get("total_events", 0),
        "origins": filtered,
        "included": {
            # Use raw totals from the aggregator; do NOT recompute here
            "flags": totals.get("flags", 0),
            "triggers": totals.get("triggers", 0),
        },
    }
    return response