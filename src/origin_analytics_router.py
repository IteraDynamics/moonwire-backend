# src/origin_analytics_router.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from pathlib import Path

from src.paths import (
    BASE_DIR,
    LOGS_DIR,
    RETRAINING_LOG_PATH as RLP_PRIMARY,
    RETRAINING_TRIGGERED_LOG_PATH as RTL_PRIMARY,
)
from src.analytics.origin_utils import compute_origin_breakdown

router = APIRouter(prefix="/internal")

def _fallback_path(primary: Path, filename: str) -> Path:
    """
    If the primary (usually temp) file is empty, fallback to default repo logs file.
    This defuses test/import-order mismatches where writers used the default path.
    """
    try:
        # primary exists but might be empty because it was created by the fixture
        if primary.exists() and primary.stat().st_size > 0:
            return primary
    except Exception:
        pass

    # Try default path under the repo logs directory
    default_dir = BASE_DIR / "logs"
    default = default_dir / filename
    try:
        if default.exists() and default.stat().st_size > 0:
            return default
    except Exception:
        pass

    # Fall back to whatever we were given; it's okay if it's empty
    return primary

@router.get("/signal-origins", summary="Origin breakdown of flags (and optional triggers)")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=0, description="Drop origins below this count"),
    include_triggers: bool = Query(True, description="Include retraining triggers"),
) -> Dict[str, Any]:
    """
    Counts and percent share for signal origins over a window.
    Reads:
      - retraining_log.jsonl (flags)
      - retraining_triggered.jsonl (triggers, optional)
    """

    # Resolve paths with safe fallback (handles test import-order mismatches)
    flags_path = _fallback_path(Path(RLP_PRIMARY), "retraining_log.jsonl")
    trig_path  = _fallback_path(Path(RTL_PRIMARY), "retraining_triggered.jsonl")

    try:
        rows, totals = compute_origin_breakdown(
            flags_path=flags_path,
            triggers_path=trig_path,
            days=days,
            include_triggers=include_triggers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"origin breakdown failed: {e}")

    if min_count > 1:
        rows = [r for r in rows if r["count"] >= min_count]

    return {
        "window_days": days,
        "total_events": totals["total_events"],
        "origins": rows,  # already sorted by utils
        "included": {"flags": totals["flags"], "triggers": totals["triggers"]},
    }