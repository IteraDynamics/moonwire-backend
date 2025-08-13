# src/origin_analytics_router.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Any, Dict
import importlib
from pathlib import Path

from src.analytics.origin_utils import compute_origin_breakdown

# NOTE:
# We DO NOT set a router-level prefix here.
# The route path below includes "/internal/...".
# In main.py, include this router WITHOUT an extra prefix to avoid "/internal/internal/...".
router = APIRouter()


def _paths():
    """
    Resolve src.paths at call time so tests that reload/monkeypatch it
    (e.g., override LOGS_DIR) are honored here.
    """
    return importlib.import_module("src.paths")


@router.get("/internal/signal-origins", summary="Origin breakdown of flags (and optional triggers)")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=0, description="Drop origins below this count"),
    include_triggers: bool = Query(True, description="Include retraining triggers in counts"),
) -> Dict[str, Any]:
    """
    Returns counts and percentage share for signal origins over the given window.

    Reads:
      - logs/retraining_log.jsonl (flags)
      - logs/retraining_triggered.jsonl (triggers, if include_triggers=True)

    The underlying utils handle:
      - epoch-second timestamps
      - origin normalization (aliases -> canonical names)
      - skipping malformed lines safely
      - sorting by count desc, then origin asc
    """
    try:
        p = _paths()
        flags_path: Path = Path(p.RETRAINING_LOG_PATH)
        trig_path: Path = Path(p.RETRAINING_TRIGGERED_LOG_PATH)

        rows, totals = compute_origin_breakdown(
            flags_path=flags_path,
            triggers_path=trig_path,
            days=days,
            include_triggers=include_triggers,
        )
    except ValueError as ve:
        # For invalid params (e.g. days <= 0) raise 400
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Keep the internal endpoint resilient and debuggable
        raise HTTPException(status_code=500, detail=f"origin breakdown failed: {e}")

    # Apply min_count AFTER aggregation; do not change totals
    if min_count > 1:
        rows = [r for r in rows if r.get("count", 0) >= min_count]

    return {
        "window_days": days,
        "total_events": totals.get("total_events", 0),
        "origins": rows,  # already sorted by utils
        "included": {
            "flags": totals.get("flags", 0),
            "triggers": totals.get("triggers", 0),
        },
    }