import json
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src.paths import RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH

CONSENSUS_THRESHOLD = 2.5

router = APIRouter(prefix="/internal")


@router.get("/consensus-debug/{signal_id}")
def consensus_debug(signal_id: str):
    log_path = Path(RETRAINING_LOG_PATH)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No retraining log found")

    # Load reviewer scores (fallback)
    scores_path = Path(REVIEWER_SCORES_PATH)
    fallback_scores = {}
    if scores_path.exists():
        with open(scores_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                fallback_scores[rec["reviewer_id"]] = rec.get("score", 1.0)

    # Parse log entries for this signal
    all_flags = []
    seen_reviewers = set()
    counted_reviewers = []
    total_weight = 0.0

    with open(log_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("signal_id") != signal_id:
                continue

            reviewer_id = entry["reviewer_id"]
            weight = entry.get("reviewer_weight")
            if weight is None:
                weight = fallback_scores.get(reviewer_id, 1.0)

            is_duplicate = reviewer_id in seen_reviewers
            all_flags.append({
                "reviewer_id": reviewer_id,
                "reviewer_weight": weight,
                "timestamp": datetime.fromtimestamp(entry["timestamp"]).isoformat(),
                "duplicate": is_duplicate
            })

            if not is_duplicate:
                seen_reviewers.add(reviewer_id)
                counted_reviewers.append(reviewer_id)
                total_weight += weight

    if not all_flags:
        raise HTTPException(status_code=404, detail="No entries for this signal_id")

    return {
        "signal_id": signal_id,
        "all_flags": all_flags,
        "counted_reviewers": counted_reviewers,
        "total_weight_used": total_weight,
        "threshold": CONSENSUS_THRESHOLD,
        "triggered": total_weight >= CONSENSUS_THRESHOLD
    }


@router.post("/evaluate-consensus-retraining")
def evaluate_consensus_retraining(payload: Dict[str, str]):
    signal_id = payload.get("signal_id")
    if not signal_id:
        raise HTTPException(status_code=400, detail="Missing signal_id")

    log_path = Path(RETRAINING_LOG_PATH)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No retraining log found")

    # Load reviewer scores (fallback)
    scores_path = Path(REVIEWER_SCORES_PATH)
    fallback_scores = {}
    if scores_path.exists():
        with open(scores_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                fallback_scores[rec["reviewer_id"]] = rec.get("score", 1.0)

    seen = set()
    total_weight = 0.0

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
            seen.add(reviewer_id)

            weight = entry.get("reviewer_weight")
            if weight is None:
                weight = fallback_scores.get(reviewer_id, 1.0)
            total_weight += weight

    return {
        "signal_id": signal_id,
        "total_weight": total_weight,
        "triggered": total_weight >= CONSENSUS_THRESHOLD,
        "threshold": CONSENSUS_THRESHOLD
    }