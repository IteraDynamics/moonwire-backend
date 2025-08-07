# src/consensus_router.py

import json
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Dict
from datetime import datetime

import src.paths as paths  # dynamic so tests can monkeypatch

CONSENSUS_THRESHOLD = 2.5

router = APIRouter(prefix="/internal")


def _score_to_weight(score):
    """Map raw reviewer score to consensus weight bands."""
    if score is None:
        return 1.0
    if score >= 0.75:
        return 1.25
    if score >= 0.5:
        return 1.0
    return 0.75


@router.get("/consensus-debug/{signal_id}")
def consensus_debug(signal_id: str):
    """
    Audit trail for a single signal. IMPORTANT: fallback uses RAW score (no banding),
    matching the Task 3 spec & test expectations.
    """
    log_path = Path(paths.RETRAINING_LOG_PATH)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No retraining log found")

    # Load reviewer scores (raw; no banding in debug)
    scores_path = Path(paths.REVIEWER_SCORES_PATH)
    raw_scores: Dict[str, float] = {}
    if scores_path.exists():
        with scores_path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                raw_scores[rec["reviewer_id"]] = rec.get("score", 1.0)

    all_flags = []
    seen_reviewers = set()
    counted_reviewers = []
    total_weight_used = 0.0

    with log_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("signal_id") != signal_id:
                continue

            reviewer_id = entry["reviewer_id"]
            # Fallback is RAW score for debug
            weight = entry.get("reviewer_weight")
            if weight is None:
                weight = raw_scores.get(reviewer_id, 1.0)

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
                total_weight_used += weight

    if not all_flags:
        raise HTTPException(status_code=404, detail="No entries for this signal_id")

    return {
        "signal_id": signal_id,
        "all_flags": all_flags,
        "counted_reviewers": counted_reviewers,
        "total_weight_used": total_weight_used,
        "threshold": CONSENSUS_THRESHOLD,
        "triggered": total_weight_used >= CONSENSUS_THRESHOLD
    }


@router.post("/evaluate-consensus-retraining")
def evaluate_consensus_retraining(payload: Dict[str, str]):
    """
    Threshold decision endpoint. IMPORTANT: fallback uses BANDING (1.25/1.0/0.75),
    matching Task 2 tests.
    """
    signal_id = payload.get("signal_id")
    if not signal_id:
        raise HTTPException(status_code=400, detail="Missing signal_id")

    log_path = Path(paths.RETRAINING_LOG_PATH)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Retraining log not found")

    # Load reviewer scores (raw), apply banding when used
    scores_path = Path(paths.REVIEWER_SCORES_PATH)
    raw_scores: Dict[str, float] = {}
    if scores_path.exists():
        with scores_path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                raw_scores[rec["reviewer_id"]] = rec.get("score")

    seen = set()
    total_weight = 0.0
    reviewer_weights: Dict[str, float] = {}

    with log_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
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
                weight = _score_to_weight(raw_scores.get(reviewer_id))

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