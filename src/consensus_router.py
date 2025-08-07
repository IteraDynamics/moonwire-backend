# src/consensus_router.py

from fastapi import APIRouter, HTTPException
from src.paths import RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH
from src.reviewer_log_utils import load_jsonl, get_reviewer_weight

router = APIRouter()

def _get_raw_score(reviewer_id: str) -> float:
    """
    Look up a reviewer’s raw score in REVIEWER_SCORES_PATH.
    Returns 0.0 if no entry is found.
    """
    for entry in load_jsonl(REVIEWER_SCORES_PATH):
        if entry.get("reviewer_id") == reviewer_id:
            return entry.get("score", 0.0)
    return 0.0

@router.get("/consensus-status/{signal_id}")
async def consensus_status(signal_id: str):
    """
    Returns how many distinct reviewers have flagged this signal_id
    for retraining, and their combined trust‐weight.
    """
    # 1) load all retraining entries
    entries = load_jsonl(RETRAINING_LOG_PATH)

    # 2) filter for this signal_id
    matched = [e for e in entries if e.get("signal_id") == signal_id]
    if not matched:
        raise HTTPException(status_code=404, detail="No retraining entries for this signal")

    # 3) build unique reviewer set + their weights
    seen: dict[str, float] = {}
    for e in matched:
        rid = e.get("reviewer_id")
        if rid in seen:
            continue

        wt = e.get("reviewer_weight")
        if wt is None:
            # first try raw score (to satisfy tests expecting raw weight)
            raw = _get_raw_score(rid)
            if raw and raw != 0.0:
                wt = raw
            else:
                # fallback to banded weight if no raw score
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