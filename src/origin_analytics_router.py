# src/origin_analytics_router.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from pathlib import Path

from src.analytics.origin_utils import compute_origin_breakdown

router = APIRouter(prefix="/internal")


@router.get("/signal-origins", summary="Origin breakdown of flags (and optional triggers)")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=0, description="Drop origins below this count"),
    include_triggers: bool = Query(True, description="Include retraining triggers"),
) -> Dict[str, Any]:
    """
    Returns counts and percentage share for signal origins over the given window.
    Reads:
      - RETRAINING_LOG_PATH (flags)
      - RETRAINING_TRIGGERED_LOG_PATH (triggers, if include_triggers=True)

    NOTE: We resolve src.paths at request-time so tests that monkeypatch/reload
    paths (e.g., isolated_logs) are respected.
    """
    try:
        # Resolve paths dynamically at request time
        from src import paths as _paths  # defer import so monkeypatch/reload takes effect

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

    # Apply min_count filter AFTER computing totals (so included counts stay correct)
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