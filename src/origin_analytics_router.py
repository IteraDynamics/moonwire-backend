from __future__ import annotations

import os
from pathlib import Path
from fastapi import APIRouter, Query

# We still import the module for fallback defaults
import src.paths as paths_mod
from src.analytics.origin_utils import compute_origin_breakdown

router = APIRouter(prefix="/internal")

def _resolve_paths() -> tuple[Path, Path]:
    """
    Resolve log file paths at request time, preferring LOGS_DIR env var
    (used by tests to isolate state). Falls back to src.paths constants.
    """
    env_logs = os.getenv("LOGS_DIR")
    if env_logs:
        base = Path(env_logs)
        return base / "retraining_log.jsonl", base / "retraining_triggered.jsonl"
    # Fallback to module constants (already Path objects in your repo)
    return Path(paths_mod.RETRAINING_LOG_PATH), Path(paths_mod.RETRAINING_TRIGGERED_LOG_PATH)

@router.get("/signal-origins")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=1, description="Drop origins with fewer than this count"),
    include_triggers: bool = Query(True, description="Include retraining triggers in counts"),
):
    """
    Origin breakdown over the window. Paths are resolved per request to respect test isolation.
    """
    flags_path, triggers_path = _resolve_paths()

    origins, totals = compute_origin_breakdown(
        flags_path,
        triggers_path,
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