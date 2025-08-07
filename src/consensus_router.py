import json
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta

from src.paths import RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH

CONSENSUS_THRESHOLD = 2.5

router = APIRouter(prefix="/internal")


@router.post("/evaluate-consensus-retraining")
def evaluate_consensus_retraining(payload: Dict[str, str]):
    signal_id = payload.get("signal_id")
    if not signal_id:
        raise HTTPException(status_code=400, detail="Missing signal_id")

    log_path = Path(RETRAINING_LOG_PATH)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Retraining log not found")

    # Load reviewer fallback scores
    fallback_scores = {}
    scores_path = Path(REVIEWER_SCORES_PATH)
    if scores_path.exists():
        with open(scores_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                fallback_scores[rec["reviewer_id"]] = rec.get("score", 1.0)

    seen = set()
    total_weight = 0.0
    reviewer_weights = {}

    with open(log_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("signal_id") != signal_id:
                continue

            reviewer_id = entry["reviewer_id"]
            if reviewer_id in seen:
                continue

            weight = entry.get("reviewer_weight")
            if weight is None:
                weight = fallback_scores.get(reviewer_id, 1.0)

            seen.add(reviewer_id)
            total_weight += weight
            reviewer_weights[reviewer_id] = weight

    return {
        "signal_id": signal_id,
        "total_weight": total_weight,
        "triggered": total_weight >= CONSENSUS_THRESHOLD,
        "threshold": CONSENSUS_THRESHOLD,
        "reviewers": [{"id": r, "weight": reviewer_weights[r]} for r in seen]
    }