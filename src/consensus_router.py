from fastapi import APIRouter, HTTPException
from src.paths import RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH
from src.reviewer_impact_scorer_router import get_reviewer_weight
from src.reviewer_log_utils import load_jsonl

router = APIRouter()

@router.get("/consensus-status/{signal_id}")
async def consensus_status(signal_id: str):
    """
    Returns how many distinct reviewers have flagged this signal_id
    for retraining, and their combined trust-weight.
    """
    # 1) load all retraining entries
    entries = load_jsonl(RETRAINING_LOG_PATH)

    # 2) filter for this signal_id
    matched = [e for e in entries if e.get("signal_id") == signal_id]
    if not matched:
        raise HTTPException(status_code=404, detail="No retraining entries for this signal")

    # 3) build unique reviewer set + their weights
    seen = {}
    for e in matched:
        rid = e.get("reviewer_id")
        if rid not in seen:
            # trust-weight stored on the log entry takes precedence
            wt = e.get("reviewer_weight")
            if wt is None:
                wt = get_reviewer_weight(rid)
            seen[rid] = wt

    reviewers = [{"reviewer_id": rid, "weight": wt} for rid, wt in seen.items()]
    total_reviewers = len(reviewers)
    combined_weight = sum(r["weight"] for r in reviewers)

    return {
        "signal_id":       signal_id,
        "total_reviewers": total_reviewers,
        "combined_weight": combined_weight,
        "reviewers":       reviewers,
    }