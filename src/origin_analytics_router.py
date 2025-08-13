# src/origin_analytics_router.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from pathlib import Path

from src.paths import RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH
from src.analytics.origin_utils import compute_origin_breakdown

router = APIRouter(prefix="/internal")


@router.get("/signal-origins", summary="Origin breakdown of flags (and optional triggers)")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=0, description="Drop origins below this count"),
    include_triggers: bool = Query(True, description="Include retraining triggers"),
) -> Dict[str, Any]:
    """
    Counts events by origin over a window.
    - Treats every line in retraining_log.jsonl as a flag.
    - Treats every line in retraining_triggered.jsonl as a trigger (if include_triggers).
    - Robust timestamp parsing and origin extraction; missing origin -> 'unknown'.
    - Sorting by count desc, then origin asc.
    """
    try:
        rows, totals = compute_origin_breakdown(
            Path(RETRAINING_LOG_PATH),
            Path(RETRAINING_TRIGGERED_LOG_PATH),
            days=days,
            include_triggers=include_triggers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"origin breakdown failed: {e}")

    # Filter AFTER computing totals (so included.* remains correct)
    if min_count > 1:
        rows = [r for r in rows if r["count"] >= min_count]

    return {
        "window_days": days,
        "total_events": totals["total_events"],
        "origins": rows,
        "included": {
            "flags": totals["flags"],
            "triggers": totals["triggers"],
        },
    }