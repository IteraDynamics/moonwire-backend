from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from src.paths import LOGS_DIR
from src.analytics.source_yield import compute_source_yield

router = APIRouter()


@router.get("/internal/source-yield-plan")
def source_yield_plan(
    days: int = Query(7, ge=1, le=30),
    min_events: int = Query(5, ge=1),
    alpha: float = Query(0.7, ge=0.0, le=1.0)
):
    flags_path = LOGS_DIR / "retraining_log.jsonl"
    triggers_path = LOGS_DIR / "retraining_triggered.jsonl"

    result = compute_source_yield(flags_path, triggers_path, days, min_events, alpha)
    return JSONResponse(content=result)