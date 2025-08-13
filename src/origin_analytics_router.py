# src/origin_analytics_router.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from pathlib import Path

from src.analytics.origin_utils import compute_origin_breakdown

router = APIRouter()

@router.get("/signal-origins", summary="Origin breakdown of flags (and optional triggers)")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=0, description="Drop origins below this count"),
    include_triggers: bool = Query(True, description="Include retraining triggers"),
) -> Dict[str, Any]:
    """
    Counts events by origin over the window.
    IMPORTANT: Resolve log paths at request time so test fixtures that reload
    src.paths (and change LOGS_DIR) are honored.
    """
    try:
        # Late import so conftest's importlib.reload(src.paths) takes effect
        from src import paths as _paths

        flags_path = Path(_paths.RETRAINING_LOG_PATH)
        triggers_path = Path(_paths.RETRAINING_TRIGGERED_LOG_PATH)

        rows, totals = compute_origin_breakdown(
            flags_path,
            triggers_path,
            days=days,
            include_triggers=include_triggers,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"origin breakdown failed: {e}")

    # Apply min_count AFTER computing totals
    if min_count > 1:
        rows = [r for r in rows if r["count"] >= min_count]

    return {
        "window_days": days,
        "total_events": totals["total_events"],
        "origins": rows,  # already sorted in utils
        "included": {
            "flags": totals["flags"],
            "triggers": totals["triggers"],
        },
    }