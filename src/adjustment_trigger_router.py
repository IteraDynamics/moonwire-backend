# src/adjustment_trigger_router.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from scripts.ml_utils.train_feedback_disagreement_model import predict_disagreement

router = APIRouter(prefix="/internal")


class RetrainRequest(BaseModel):
    signal_id: str
    reason:    str
    note:      Optional[str] = None


@router.post("/flag-for-retraining", status_code=200)
async def flag_for_retraining(req: RetrainRequest):
    """
    Records a signal for later retraining.
    """
    # TODO: hook into your retraining queue here
    return {"retrain_queued": True, "signal_id": req.signal_id}


class OverrideRequest(BaseModel):
    signal_id:        str
    override_reason:  str
    note:             Optional[str] = None
    reviewed_by:      Optional[str] = None


@router.post("/override-suppression", status_code=200)
async def override_suppression(req: OverrideRequest):
    """
    Applies a manual override to a suppressed signal.
    """
    # TODO: implement real override logic (e.g. record to JSONL)
    return {"override_applied": True, "signal_id": req.signal_id}


@router.post("/adjust-signals-based-on-feedback", status_code=200)
async def trigger_adjust_signals():
    """
    Runs the feedback‐based adjustment model.
    """
    result = predict_disagreement()
    return {"adjustment_result": result}
