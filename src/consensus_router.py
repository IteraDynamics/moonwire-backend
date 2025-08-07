# src/consensus_router.py

from fastapi import APIRouter, HTTPException
from src.paths import RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH
from src.reviewer_log_utils import load_jsonl, get_reviewer_weight

router = APIRouter()

def _get_all_scores() -> list[dict]:
    """
    Load all raw score entries (reviewer_id + score) from the scores file.
    """
    if not REVIEWER_SCORES_PATH.exists():
        return []
    return load_jsonl(REVIEWER_SCORES_PATH)

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

    # 3) load scores just once
    all_scores = _get_all_scores()
    has_any_scores = len(all_scores) > 0

    # 4) build unique reviewer set + their weights
    seen: dict[str, float] = {}
    for e in matched:
        rid = e.get("reviewer_id")
        if rid in seen:
            continue

        # trust-weight explicitly stored on the log entry takes precedence
        wt = e.get("reviewer_weight")
        if wt is None:
            # if no scores at all, default to 1.0
            if not has_any_scores:
                wt = 1.0
            else:
                # try to find a raw score for this reviewer
                raw_match = next((r for r in all_scores if r.get("reviewer_id") == rid), None)
                if raw_match:
                    wt = raw_match.get("score", 1.0)
                else:
                    # no entry for this reviewer but scores exist → use banded weight
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