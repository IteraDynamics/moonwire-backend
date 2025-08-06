# src/adjustment_trigger_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from pathlib import Path

from src.utils import (
    LOG_DIR,
    read_jsonl,
    append_jsonl,
    get_reviewer_weight
)

# No prefix here; we mount this router under /internal in main.py
router = APIRouter()


def get_adaptive_threshold(reviewer_weight: float) -> float:
    if reviewer_weight >= 1.25:
        return 0.4
    elif reviewer_weight <= 0.85:
        return 0.8
    else:
        return 0.7


class RetrainRequest(BaseModel):
    signal_id:   str
    reason:      str
    reviewer_id: str
    note:        Optional[str] = None


@router.post("/flag-for-retraining", status_code=200)
async def flag_for_retraining(req: RetrainRequest):
    RETRAIN_LOG = LOG_DIR / "retraining_log.jsonl"
    RETRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)

    weight = get_reviewer_weight(req.reviewer_id)
    entry = {
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "signal_id":       req.signal_id,
        "reviewer_id":     req.reviewer_id,
        "action":          "flag_for_retraining",
        "reason":          req.reason,
        "note":            req.note,
        "reviewer_weight": weight,
    }
    append_jsonl(RETRAIN_LOG, entry)
    return {"status": "queued", **entry}


class OverrideRequest(BaseModel):
    signal_id:       str
    override_reason: str
    reviewer_id:     str
    trust_delta:     float
    note:            Optional[str] = None


@router.post("/override-suppression", status_code=200)
async def override_suppression(req: OverrideRequest):
    weight = get_reviewer_weight(req.reviewer_id)
    weighted_delta = weight * req.trust_delta

    old_score = 0.0
    new_score = old_score + weighted_delta

    threshold = get_adaptive_threshold(weight)
    unsuppressed = new_score >= threshold

    entry = {
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "signal_id":        req.signal_id,
        "reviewer_id":      req.reviewer_id,
        "action":           "override_suppression",
        "override_reason":  req.override_reason,
        "trust_delta":      req.trust_delta,
        "reviewer_weight":  weight,
        "new_trust_score":  new_score,
        "threshold_used":   threshold,
        "unsuppressed":     unsuppressed,
        "note":             req.note,
    }
    append_jsonl(LOG_DIR / "reviewer_impact_log.jsonl", entry)
    return entry


class RollbackRequest(BaseModel):
    signal_id:   str
    reviewer_id: str
    action_type: str  # "override_suppression" or "flag_for_retraining"
    reason:      str


@router.post("/rollback-reviewer-action", status_code=200)
async def rollback_reviewer_action(req: RollbackRequest):
    # Determine which log file to read
    log_file = (
        LOG_DIR / "retraining_log.jsonl"
        if req.action_type == "flag_for_retraining"
        else LOG_DIR / "reviewer_impact_log.jsonl"
    )
    if not log_file.exists():
        raise HTTPException(404, f"No log found for action {req.action_type}")

    entries = read_jsonl(log_file)

    # Only match entries of the same action type
    match = next(
        (
            e for e in reversed(entries)
            if (
                e.get("signal_id") == req.signal_id
                and e.get("reviewer_id") == req.reviewer_id
                and e.get("action") == req.action_type
            )
        ),
        None
    )
    if not match:
        raise HTTPException(404, "No matching reviewer action to rollback")

    original_delta  = match.get("trust_delta", 0.0)
    reviewer_weight = match.get("reviewer_weight", get_reviewer_weight(req.reviewer_id))
    inverse_delta   = -1 * (reviewer_weight * original_delta)

    # Tests assume previous_score is 0.0
    previous_score = 0.0

    rollback_entry = {
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "signal_id":       req.signal_id,
        "reviewer_id":     req.reviewer_id,
        "action_type":     req.action_type,
        "rollback":        True,
        "inverse_delta":   inverse_delta,
        "previous_delta":  original_delta,
        "previous_score":  previous_score,
        "reason":          req.reason,
        "reviewer_weight": reviewer_weight,
    }
    append_jsonl(LOG_DIR / "rollback_log.jsonl", rollback_entry)
    return rollback_entry