#!/usr/bin/env python3
"""
Router: /internal/source-yield-plan

Computes per-origin yield score and returns suggested API budget allocation.
"""

from fastapi import APIRouter, Query
from pathlib import Path

from .analytics.source_yield import compute_source_yield
import src.paths as paths  # your central path definitions

router = APIRouter()


@router.get("/internal/source-yield-plan")
def get_source_yield_plan(
    days: int = Query(7, ge=1, le=90, description="Lookback window in days"),
    min_events: int = Query(5, ge=1, description="Minimum flags required to be eligible"),
    alpha: float = Query(0.7, ge=0.0, le=1.0, description="Weight between trigger_rate and volume_share")
):
    """
    Returns per-origin yield score and budget allocation plan.

    Scoring:
        yield_score = alpha * trigger_rate + (1 - alpha) * volume_share
        trigger_rate = triggers / flags
        volume_share = flags / total_flags
    """
    flags_path = paths.RETRAINING_LOG_PATH
    triggers_path = paths.RETRAINING_TRIGGERED_LOG_PATH

    result = compute_source_yield(
        flags_path=flags_path,
        triggers_path=triggers_path,
        days=days,
        min_events=min_events,
        alpha=alpha
    )
    return result
