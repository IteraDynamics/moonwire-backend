# src/origin_analytics_router.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
from pathlib import Path

# IMPORTANT: import the module, not the values
import src.paths as paths
from src.analytics.origin_utils import compute_origin_breakdown

router = APIRouter(prefix="/internal")

# Allow tests to monkeypatch these if they want to point at tmp files.
# If left as None, we’ll fall back to paths.RETRAINING_LOG_PATH* at call time.
FLAGS_PATH_OVERRIDE: Optional[Path] = None
TRIGGERS_PATH_OVERRIDE: Optional[Path] = None

def _resolve_flags_path() -> Path:
    return FLAGS_PATH_OVERRIDE or paths.RETRAINING_LOG_PATH

def _resolve_triggers_path() -> Path:
    return TRIGGERS_PATH_OVERRIDE or paths.RETRAINING_TRIGGERED_LOG_PATH

@router.get("/signal-origins", summary="Origin breakdown of flags (and optional triggers)")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=0, description="Drop origins below this count"),
    include_triggers: bool = Query(True, description="Include retraining triggers"),
) -> Dict[str, Any]:
    """
    Returns counts and % share per origin for the given window.
    Reads:
      - RETRAINING_LOG_PATH (flags)
      - RETRAINING_TRIGGERED_LOG_PATH (triggers, if include_triggers=True)
    """
    try:
        rows, totals = compute_origin_breakdown(
            _resolve_flags_path(),
            _resolve_triggers_path(),
            days=days,
            include_triggers=include_triggers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"origin breakdown failed: {e}")

    # Apply min_count AFTER aggregation (does not change totals)
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
