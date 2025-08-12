from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone

from src.paths import RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH
from src.analytics.origin_utils import compute_origin_breakdown, normalize_origin

router = APIRouter(prefix="/internal")

@router.get("/signal-origins")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=1, description="Drop origins with fewer than this count"),
    include_triggers: bool = Query(True, description="Include retraining triggers in counts"),
):
    """
    Return origin breakdown over the window.
    """
    origins, totals = compute_origin_breakdown(
        Path(RETRAINING_LOG_PATH),
        Path(RETRAINING_TRIGGERED_LOG_PATH),
        days=days,
        include_triggers=include_triggers,
    )

    # Apply min_count filter AFTER computing totals/pct
    filtered = [o for o in origins if o["count"] >= min_count]

    return {
        "window_days": days,
        "total_events": totals.get("total_events", 0),
        "origins": filtered,
        "included": {
            "flags": totals.get("flags", 0),
            "triggers": totals.get("triggers", 0) if include_triggers else 0,
        },
    }