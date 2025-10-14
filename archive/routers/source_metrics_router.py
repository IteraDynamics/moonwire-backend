from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import os

from src.paths import LOGS_DIR
from src.analytics.source_metrics import compute_source_metrics

router = APIRouter()

@router.get("/internal/source-metrics")
def source_metrics(
    days: int = Query(7, ge=1, le=30),
    min_count: int = Query(1, ge=0)
):
    try:
        result = compute_source_metrics(
            flags_path=LOGS_DIR / "retraining_log.jsonl",
            triggers_path=LOGS_DIR / "retraining_triggered.jsonl",
            days=days,
            min_count=min_count
        )
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to compute source metrics", "detail": str(e)}
        )
