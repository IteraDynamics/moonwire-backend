from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
import importlib

import src.paths as _paths  # module, not constants
from src.analytics.origin_utils import compute_origin_breakdown

router = APIRouter(prefix="/internal")

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

    # IMPORTANT: Re-resolve paths at request time so tests that monkeypatch LOGS_DIR
    # (and reload src.paths) are respected here, avoiding stale import-time constants.
    paths = importlib.reload(_paths)
    flags_path = paths.RETRAINING_LOG_PATH
    trig_path = paths.RETRAINING_TRIGGERED_LOG_PATH

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