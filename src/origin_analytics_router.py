# src/origin_analytics_router.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from pathlib import Path

from src.analytics.origin_utils import compute_origin_breakdown

router = APIRouter(prefix="/internal")

def _resolve_paths() -> tuple[Path, Path]:
    """
    Resolve current log paths at request time.
    This avoids freezing old paths that were imported before tests
    changed LOGS_DIR and reloaded src.paths.
    """
    from src import paths as P  # import inside to pick up reloaded module
    return Path(P.RETRAINING_LOG_PATH), Path(P.RETRAINING_TRIGGERED_LOG_PATH)

@router.get("/signal-origins", summary="Origin breakdown of flags (and optional triggers)")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=0, description="Drop origins below this count"),
    include_triggers: bool = Query(True, description="Include retraining triggers"),
) -> Dict[str, Any]:
    """
    Returns counts and percentage share for signal origins over the given window.
    Reads:
      - retraining_log.jsonl (flags) — always counted
      - retraining_triggered.jsonl (triggers) — added when include_triggers=True
    """
    try:
        flags_path, triggers_path = _resolve_paths()
        rows, totals = compute_origin_breakdown(
            flags_path,
            triggers_path,
            days=days,
            include_triggers=include_triggers,
        )
    except ValueError as ve:
        # e.g., days <= 0
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"origin breakdown failed: {e}")

    # Apply min_count AFTER aggregating so totals stay correct
    if min_count > 1:
        rows = [r for r in rows if r["count"] >= min_count]

    return {
        "window_days": days,
        "total_events": totals["total_events"],
        "origins": rows,  # already sorted in utils by count desc, origin asc
        "included": {
            "flags": totals["flags"],
            "triggers": totals["triggers"],
        },
    }