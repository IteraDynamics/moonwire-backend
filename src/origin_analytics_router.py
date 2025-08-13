# src/origin_analytics_router.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

# IMPORTANT: import the module, not the constants, so pytest's monkeypatch/reload
# in tests (isolated_logs) is respected at request time.
import src.paths as paths
from src.analytics.origin_utils import compute_origin_breakdown

# Keep this prefix if main.py mounts without an extra "/internal"
router = APIRouter(prefix="/internal")


@router.get("/signal-origins", summary="Origin breakdown of flags (and optional triggers)")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=0, description="Drop origins below this count"),
    include_triggers: bool = Query(True, description="Include retraining triggers"),
) -> Dict[str, Any]:
    """
    Returns counts and percentage share for signal origins over the given window.

    Reads (streamed line-by-line):
      - paths.RETRAINING_LOG_PATH (flags)
      - paths.RETRAINING_TRIGGERED_LOG_PATH (triggers, if include_triggers=True)
    """

    try:
        # Resolve paths at REQUEST TIME (critical for tests using isolated temp dirs)
        flags_path = Path(paths.RETRAINING_LOG_PATH)
        trig_path = Path(paths.RETRAINING_TRIGGERED_LOG_PATH)

        # --- Temporary debug aid (uncomment locally if needed) ---
        print(
        f"[origin] using flags_path={flags_path} exists={flags_path.exists()} "
        f"lines={(sum(1 for _ in flags_path.open()) if flags_path.exists() else 0)}; "
        f"trig_path={trig_path} exists={trig_path.exists()} "
        f"lines={(sum(1 for _ in trig_path.open()) if trig_path.exists() else 0)}"
        )

        rows, totals = compute_origin_breakdown(
            flags_path=flags_path,
            triggers_path=trig_path,
            days=days,
            include_triggers=include_triggers,
        )
    except HTTPException:
        # bubble up explicit 4xx
        raise
    except Exception as e:
        # Keep internal endpoint resilient
        raise HTTPException(status_code=500, detail=f"origin breakdown failed: {e}")

    # Apply min_count filter AFTER aggregating (do not alter totals)
    if min_count > 1:
        rows = [r for r in rows if r.get("count", 0) >= min_count]

    # rows are already sorted in utils (count desc, origin asc)
    return {
        "window_days": days,
        "total_events": totals.get("total_events", 0),
        "origins": rows,
        "included": {
            "flags": totals.get("flags", 0),
            "triggers": totals.get("triggers", 0),
        },
    }