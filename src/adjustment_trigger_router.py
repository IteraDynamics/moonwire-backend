# src/adjustment_trigger_router.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from scripts.ml_utils.train_feedback_disagreement_model import predict_disagreement

router = APIRouter()


class RetrainRequest(BaseModel):
    signal_id: str
    reason: str
    note: Optional[str] = None


@router.post("/internal/flag-for-retraining", status_code=200, summary="Flag a signal for retraining")
async def flag_for_retraining(req: RetrainRequest):
    """
    Stub endpoint to record a signal for later retraining.
    """
    # TODO: replace with actual retraining-queue logic
    return {"retrain_queued": True, "signal_id": req.signal_id}


@router.post("/internal/adjust-signals-based-on-feedback", summary="Trigger feedback‐based signal adjustment")
def trigger_adjust_signals():
    """
    Existing endpoint — runs the feedback disagreement prediction logic.
    """
    result = predict_disagreement()
    return {"adjustment_result": result}