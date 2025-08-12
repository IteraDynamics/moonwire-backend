# src/origin_analytics_router.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pathlib import Path

from src.paths import (
    RETRAINING_LOG_PATH,
    RETRAINING_TRIGGERED_LOG_PATH,
)
from src.analytics.origin_utils import compute_origin_breakdown

router = APIRouter(prefix="/internal")


@router.get("/signal-origins")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=1, description="Drop origins with fewer than this count"),
    include_triggers: bool = Query(True, description="Include retraining triggers in counts"),
):
    """
    Aggregate flag/trigger events by origin over the given window.
    - Streams JSONL files
    - Normalizes origins (e.g., twitter_api -> twitter)
    - Computes counts and percent share
    """
    # days <= 0 gets blocked by Query(ge=1), but keep a guard in case
    if days <= 0:
        raise HTTPException(status_code=400, detail="days must be >= 1")

    # Compute breakdown
    origins_list, totals = compute_origin_breakdown(
        flags_path=Path(RETRAINING_LOG_PATH),
        triggers_path=Path(RETRAINING_TRIGGERED_LOG_PATH) if include_triggers else None,
        days=days,
        include_triggers=include_triggers,
    )

    # Apply min_count filter AFTER counting
    filtered = [o for o in origins_list if o["count"] >= min_count]

    return {
        "window_days": days,
        "total_events": totals["total_events"],
        "origins": filtered,
        "included": {
            "flags": totals["flags"],
            "triggers": totals["triggers"],
        },
    }