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
    Audit trail for a single signal. Fallback uses RAW score (no banding).
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
            weight = entry.get("reviewer_weight")
            if weight is None:
                weight = raw_scores.get(reviewer_id, 1.0)  # RAW for debug

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
    Threshold decision endpoint. Fallback uses BANDING (1.25/1.0/0.75).
    Also writes a trigger log line when threshold is met.
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

    triggered = total_weight >= CONSENSUS_THRESHOLD

    # 🔐 Write trigger log if threshold met
    if triggered:
        trig_path = Path(paths.RETRAINING_TRIGGERED_LOG_PATH)
        trig_path.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "signal_id": signal_id,
            "total_weight": total_weight,
            "threshold": CONSENSUS_THRESHOLD,
            "reviewers": [{"id": r, "weight": reviewer_weights[r]} for r in seen],
            "timestamp": datetime.utcnow().isoformat()
        }
        with trig_path.open("a") as f:
            f.write(json.dumps(log_entry) + "\n")

    return {
        "signal_id": signal_id,
        "total_weight": total_weight,
        "triggered": triggered,
        "threshold": CONSENSUS_THRESHOLD,
        "reviewers": [{"id": r, "weight": reviewer_weights[r]} for r in seen]
    }