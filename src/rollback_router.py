# src/rollback_router.py

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal

from src.paths import REVIEWER_IMPACT_LOG_PATH
from src.reviewer_log_utils import load_jsonl, append_jsonl, get_reviewer_weight, apply_trust_delta

router = APIRouter()

class RollbackRequest(BaseModel):
    signal_id:   str
    reviewer_id: str
    action_type: Literal["override_suppression", "flag_for_retraining"]
    reason:      str

@router.post("/rollback-reviewer-action")
async def rollback_reviewer_action(req: RollbackRequest):
    # 1) load all impact entries
    entries = load_jsonl(REVIEWER_IMPACT_LOG_PATH)
    # 2) find the most recent matching entry
    match = None
    for e in reversed(entries):
        if (
            e.get("signal_id") == req.signal_id
            and e.get("reviewer_id") == req.reviewer_id
            and e.get("action") == req.action_type
        ):
            match = e
            break

    if not match:
        raise HTTPException(404, "No matching reviewer action to roll back")

    # 3) compute inverse delta
    orig_delta  = match["trust_delta"]
    weight      = match.get("reviewer_weight", get_reviewer_weight(req.reviewer_id))
    inverse     = -1 * (orig_delta * weight)

    # 4) apply it (using your existing helper that updates scores)
    previous, new = apply_trust_delta(req.signal_id, inverse)

    # 5) log the rollback
    append_jsonl(REVIEWER_IMPACT_LOG_PATH, {
        "signal_id":         req.signal_id,
        "reviewer_id":       req.reviewer_id,
        "action":            req.action_type,
        "trust_delta":       inverse,
        "reviewer_weight":   weight,
        "rollback":          True,
        "reason":            req.reason,
        "timestamp":         time.time(),
        "previous_score":    previous,
        "new_score":         new,
    })

    return {
        "inverse_delta":   inverse,
        "previous_score":  previous,
        "new_score":       new,
    }