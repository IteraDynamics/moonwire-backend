# src/adjustment_trigger_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path

from scripts.ml_utils.train_feedback_disagreement_model import predict_disagreement

# Path to the reviewer scores JSONL file
REVIEWER_SCORES_PATH = Path(__file__).resolve().parent.parent / "logs" / "reviewer_scores.jsonl"

router = APIRouter(prefix="/internal")


class RetrainRequest(BaseModel):
    signal_id: str
    reason: str
    note: Optional[str] = None


@router.post("/flag-for-retraining", status_code=200)
async def flag_for_retraining(req: RetrainRequest):
    """
    Records a signal for later retraining.
    """
    # TODO: hook into your retraining queue here
    return {"retrain_queued": True, "signal_id": req.signal_id}


class OverrideRequest(BaseModel):
    signal_id: str
    override_reason: str
    reviewer_id: str
    trust_delta: float
    note: Optional[str] = None


@router.post("/override-suppression", status_code=200)
async def override_suppression(req: OverrideRequest):
    """
    Applies a manual override to a suppressed signal, weighting
    the trust_delta by reviewer_weight and auto-unsuppressing
    if the new trust score crosses the suppression threshold.
    """
    # 1) Load the reviewer’s current score
    score = 0.0
    if REVIEWER_SCORES_PATH.exists():
        try:
            with REVIEWER_SCORES_PATH.open("r") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("reviewer_id") == req.reviewer_id:
                        score = entry.get("score", 0.0)
                        break
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load reviewer scores: {e}")

    # 2) Compute weight multiplier
    if score >= 0.75:
        weight = 1.25
    elif score >= 0.5:
        weight = 1.0
    else:
        weight = 0.75

    # 3) Apply weighted delta
    # TODO: replace old_score=0.0 stub with real lookup (e.g. from signal history)
    old_score = 0.0
    new_score = old_score + weight * req.trust_delta

    # 4) Determine unsuppression
    suppression_threshold = 0.4
    unsuppressed = new_score >= suppression_threshold

    # 5) Debug/log output
    print(f"🚨 /internal/override-suppression hit")
    print(f"  signal_id: {req.signal_id}")
    print(f"  reviewer_id: {req.reviewer_id}, score: {score}, weight: {weight}")
    print(f"  old_score: {old_score}, trust_delta: {req.trust_delta}, new_score: {new_score}")
    print(f"  unsuppressed: {unsuppressed}")

    return {
        "override_applied": True,
        "signal_id": req.signal_id,
        "reviewer_id": req.reviewer_id,
        "reviewer_weight": weight,
        "new_trust_score": new_score,
        "unsuppressed": unsuppressed,
    }


@router.post("/adjust-signals-based-on-feedback", status_code=200)
async def trigger_adjust_signals():
    """
    Runs the feedback-based adjustment model.
    """
    result = predict_disagreement()
    return {"adjustment_result": result}