# src/consensus_router.py

import json
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List, Dict

from src.paths import LOGS_DIR, RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH

router = APIRouter(prefix="/internal")

@router.get("/consensus-status/{signal_id}", status_code=200)
def get_consensus_status(signal_id: str):
    """
    Returns how many unique reviewers have flagged a signal, 
    their weights, and the combined total.
    """
    log_path = Path(RETRAINING_LOG_PATH)
    if not log_path.exists():
        raise HTTPException(404, detail="No retraining log found")

    # 1) Read retraining entries
    reviewers: Dict[str, float] = {}
    with open(log_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("signal_id") != signal_id:
                continue
            rid = entry["reviewer_id"]
            reviewers[rid] = entry.get("reviewer_weight", None)  # we'll fill fallback next

    if not reviewers:
        raise HTTPException(404, detail="No entries for this signal_id")

    # 2) Load reviewer scores for fallback
    scores_path = Path(REVIEWER_SCORES_PATH)
    scores = {}
    if scores_path.exists():
        with open(scores_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                scores[rec["reviewer_id"]] = rec.get("score", 1.0)

    # 3) Build response list, applying fallback=1.0
    resp_reviewers: List[Dict[str, float]] = []
    total_weight = 0.0
    for rid, weight in reviewers.items():
        w = weight if weight is not None else scores.get(rid, 1.0)
        resp_reviewers.append({"reviewer_id": rid, "weight": w})
        total_weight += w

    return {
        "signal_id":       signal_id,
        "total_reviewers": len(resp_reviewers),
        "combined_weight": total_weight,
        "reviewers":       resp_reviewers,
    }
