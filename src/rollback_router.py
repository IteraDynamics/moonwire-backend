from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from src.paths import REVIEWER_IMPACT_LOG_PATH
from src.reviewer_log_utils import load_jsonl, append_jsonl, get_reviewer_weight

router = APIRouter()


@router.post("/rollback-reviewer-action", status_code=200)
async def rollback_reviewer_action(payload: Dict[str, Any]):
    """
    Reverse the effect of a previous reviewer action (override or flag-for-retraining).
    """
    signal_id = payload.get("signal_id")
    reviewer_id = payload.get("reviewer_id")
    action_type = payload.get("action_type")
    reason = payload.get("reason")

    # 1) load all impact log entries
    entries = load_jsonl(REVIEWER_IMPACT_LOG_PATH)

    # 2) find the most recent matching entry
    matched = [
        e
        for e in entries
        if e.get("signal_id") == signal_id
        and e.get("reviewer_id") == reviewer_id
        and e.get("action") == action_type
    ]
    if not matched:
        raise HTTPException(status_code=404, detail="No matching entry to rollback")

    orig = matched[-1]  # last‐in

    # 3) compute inverse delta
    raw_delta = orig.get("trust_delta", 0.0)
    reviewer_weight = orig.get("reviewer_weight")
    if reviewer_weight is None:
        reviewer_weight = get_reviewer_weight(reviewer_id)
    inverse_delta = -1 * (reviewer_weight * raw_delta)

    # 4) assume previous_score is zero if not logged
    previous_score = orig.get("previous_score", 0.0)
    new_score = previous_score + inverse_delta

    # 5) append rollback record
    record = {
        "signal_id":      signal_id,
        "reviewer_id":    reviewer_id,
        "action":         action_type,
        "reason":         reason,
        "rollback":       True,
        "inverse_delta":  inverse_delta,
        "previous_score": previous_score,
        "new_score":      new_score,
        # timestamp will be added by your log util if you want
    }
    append_jsonl(REVIEWER_IMPACT_LOG_PATH, record)

    return record