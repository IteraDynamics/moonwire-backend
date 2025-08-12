# src/origin_analytics_router.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pathlib import Path

from src.paths import RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH
from src.analytics.origin_utils import aggregate_origins, build_breakdown

router = APIRouter(prefix="/internal")

@router.get("/signal-origins")
def signal_origins(
    days: int = Query(7, ge=1, description="Lookback window in days"),
    min_count: int = Query(1, ge=1, description="Drop origins below this count"),
    include_triggers: bool = Query(True, description="Include retraining triggers"),
):
    # guard
    if days <= 0:
        raise HTTPException(status_code=400, detail="days must be > 0")

    counts, flags_included, triggers_included = aggregate_origins(
        Path(RETRAINING_LOG_PATH),
        Path(RETRAINING_TRIGGERED_LOG_PATH),
        days=days,
        include_triggers=include_triggers,
    )

    # If include_triggers=false we still pass triggers=0
    if not include_triggers:
        triggers_included = 0

    return build_breakdown(
        counts=counts,
        flags_included=flags_included,
        triggers_included=triggers_included,
        min_count=min_count,
        window_days=days,
    )

# Optional: tiny plotting payload for dashboards
@router.get("/signal-origins/chart")
def signal_origins_chart(
    days: int = Query(7, ge=1),
    min_count: int = Query(1, ge=1),
    include_triggers: bool = Query(True),
):
    counts, flags_included, triggers_included = aggregate_origins(
        Path(RETRAINING_LOG_PATH),
        Path(RETRAINING_TRIGGERED_LOG_PATH),
        days=days,
        include_triggers=include_triggers,
    )
    breakdown = build_breakdown(
        counts=counts,
        flags_included=flags_included,
        triggers_included=(triggers_included if include_triggers else 0),
        min_count=min_count,
        window_days=days,
    )
    labels = [row["origin"] for row in breakdown["origins"]]
    values = [row["count"] for row in breakdown["origins"]]
    return {
        "window_days": breakdown["window_days"],
        "labels": labels,
        "counts": values,
        "included": breakdown["included"],
        "total_events": breakdown["total_events"],
    }