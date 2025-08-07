# src/rollback_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from time import time

from src.reviewer_log_utils import (
    load_jsonl,
    get_signal_trust,
    apply_trust_delta,
    append_jsonl,
)
from src.paths import REVIEWER_IMPACT_LOG_PATH

router = APIRouter(prefix="/internal")

class RollbackRequest(BaseModel):
    signal_id:   str
    reviewer_id: str
    action_type: str            # e.g. "override_suppression" or "flag_for_retraining"
    reason:      str
    note:        Optional[str] = None

@router.post("/rollback-reviewer-action", status_code=200)
async def rollback_reviewer_action(req: RollbackRequest):
    # find last matching entry in the impact log
    entries = load_jsonl(REVIEWER_IMPACT_LOG_PATH)
    matching = [
        e for e in entries
        if e.get("signal_id") == req.signal_id
        and e.get("reviewer_id") == req.reviewer_id
        and e.get("action") == req.action_type
    ]
    if not matching:
        raise HTTPException(status_code=404, detail="No matching entry to rollback")

    orig = matching[-1]
    inverse_delta = -1 * (orig["trust_delta"] * orig["reviewer_weight"])

    old_score = get_signal_trust(req.signal_id)
    apply_trust_delta(req.signal_id, inverse_delta)
    new_score = old_score + inverse_delta

    # record the rollback as its own log entry
    payload = {
        "signal_id":        req.signal_id,
        "reviewer_id":      req.reviewer_id,
        "action":           req.action_type,
        "inverse_delta":    inverse_delta,
        "previous_score":   old_score,
        "timestamp":        time(),
        "rollback":         True,
        "reason":           req.reason,
        "note":             req.note,
    }
    append_jsonl(REVIEWER_IMPACT_LOG_PATH, payload)

    return {
        "signal_id":       req.signal_id,
        "inverse_delta":   inverse_delta,
        "previous_score":  old_score,
        "new_trust_score": new_score,
        "rollback":        True,
    }