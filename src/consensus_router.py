# src/consensus_router.py
from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

from src.paths import RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH

router = APIRouter(prefix="/internal")

def load_jsonl(path: Path):
    """Yield each line parsed as JSON dict, or empty if file missing."""
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]

def get_reviewer_weight(reviewer_id: str) -> float:
    """Lookup weight in reviewer_scores.jsonl, default to 1.0."""
    if REVIEWER_SCORES_PATH.exists():
        for entry in load_jsonl(REVIEWER_SCORES_PATH):
            if entry.get("reviewer_id") == reviewer_id:
                return entry.get("score", 1.0)
    return 1.0

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
            seen[rid] = e.get("reviewer_weight", get_reviewer_weight(rid))

    reviewers = [
        {"reviewer_id": rid, "weight": wt}
        for rid, wt in seen.items()
    ]

    total_reviewers = len(reviewers)
    combined_weight = sum(r["weight"] for r in reviewers)

    return {
        "signal_id":       signal_id,
        "total_reviewers": total_reviewers,
        "combined_weight": combined_weight,
        "reviewers":       reviewers,
    }