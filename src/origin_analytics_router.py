# src/origin_analytics_router.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Query

from src.paths import RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH
from src.analytics.origin_utils import compute_origin_breakdown

# NOTE: No prefix here; main.py should mount with prefix="/internal"
router = APIRouter()

@router.get("/signal-origins", summary="Origin breakdown of flags (and optional triggers)")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=0, description="Drop origins below this count (after counting)"),
    include_triggers: bool = Query(True, description="If true, include retraining triggers in breakdown"),
) -> Dict[str, Any]:
    try:
        rows, totals = compute_origin_breakdown(
            Path(RETRAINING_LOG_PATH),
            Path(RETRAINING_TRIGGERED_LOG_PATH),
            days=days,
            include_triggers=include_triggers,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"origin breakdown failed: {e}")

    # Apply min_count filter AFTER aggregation; do not change totals
    if min_count > 1:
        rows = [r for r in rows if r["count"] >= min_count]

    return {
        "window_days": days,
        "total_events": totals["total_events"],
        "origins": rows,
        "included": {"flags": totals["flags"], "triggers": totals["triggers"]},
    }