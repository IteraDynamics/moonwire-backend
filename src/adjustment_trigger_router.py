# src/adjustment_trigger_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from pathlib import Path
import json
import time

from scripts.ml_utils.train_feedback_disagreement_model import predict_disagreement
from src.utils import get_reviewer_weight, read_jsonl, append_jsonl, LOG_DIR

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
    weight = get_reviewer_weight(req.reviewer_id)  # default=1.0
    entry = {
        "signal_id": req.signal_id,
        "reviewer_id": req.reviewer_id,
        "action": "flag_for_retraining",
        "reason": req.reason,
        "reviewer_weight": weight,
        "timestamp": time.time(),
    }
    append_jsonl(LOG_DIR / "retraining_log.jsonl", entry)
    return entry


class OverrideRequest(BaseModel):
    signal_id:       str
    override_reason: str
    reviewer_id:     str
    trust_delta:     float


@router.post("/override-suppression", status_code=200)
async def override_suppression(req: OverrideRequest):
    """
    Applies a manual override to a suppressed signal.
    """
    weight = get_reviewer_weight(req.reviewer_id)
    weighted_delta = weight * req.trust_delta

    # read old_score from wherever you store it; default 0.0
    old_score = 0.0

    new_score = old_score + weighted_delta
    threshold = 0.4  # or use get_adaptive_threshold(weight)

    unsuppressed = new_score >= threshold

    entry = {
        "signal_id": req.signal_id,
        "reviewer_id": req.reviewer_id,
        "action": "override_suppression",
        "override_reason": req.override_reason,
        "trust_delta": req.trust_delta,
        "reviewer_weight": weight,
        "new_trust_score": new_score,
        "unsuppressed": unsuppressed,
        "timestamp": time.time(),
    }
    append_jsonl(LOG_DIR / "reviewer_impact_log.jsonl", entry)
    return entry


class RollbackRequest(BaseModel):
    signal_id:   str
    reviewer_id: str
    action_type: Literal["override_suppression", "flag_for_retraining"]
    reason:      str


@router.post("/rollback-reviewer-action", status_code=200)
async def rollback_reviewer_action(req: RollbackRequest):
    """
    Reverses the effect of a past reviewer action.
    """
    log_path = LOG_DIR / "reviewer_impact_log.jsonl"
    if not log_path.exists():
        raise HTTPException(404, "No impact log found")

    # find the most recent matching entry
    entries = read_jsonl(log_path)
    match = next(
        (e for e in reversed(entries)
         if e["signal_id"] == req.signal_id
         and e["reviewer_id"] == req.reviewer_id
         and e["action"] == req.action_type),
        None
    )
    if not match:
        raise HTTPException(404, "No matching action to rollback")

    original_delta = match.get("trust_delta", 0.0)
    weight = match.get("reviewer_weight", 1.0)
    inverse_delta = -1 * (weight * original_delta)

    # again, pull current_score from your store; default 0.0
    current_score = 0.0
    new_score = current_score + inverse_delta

    rollback_entry = {
        "signal_id": req.signal_id,
        "reviewer_id": req.reviewer_id,
        "action": req.action_type,
        "rollback": True,
        "inverse_delta": inverse_delta,
        "previous_score": current_score,
        "new_score": new_score,
        "reason": req.reason,
        "timestamp": time.time(),
    }
    append_jsonl(LOG_DIR / "rollback_log.jsonl", rollback_entry)
    return rollback_entry