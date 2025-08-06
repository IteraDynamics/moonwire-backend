# src/adjustment_trigger_router.py

import os
import json
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.paths import LOGS_DIR, RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH

router = APIRouter(prefix="/internal")


# --- Models ------------------------------------------

class RetrainRequest(BaseModel):
    signal_id:   str
    reviewer_id: Optional[str] = None
    reason:      str
    note:        Optional[str] = None


class OverrideRequest(BaseModel):
    signal_id:        str
    override_reason:  str
    reviewer_id:      Optional[str] = None
    trust_delta:      float
    note:             Optional[str] = None


# --- Helpers -----------------------------------------

def load_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_reviewer_weight(reviewer_id: str) -> float:
    """
    Look up a reviewer's raw score from reviewer_scores.jsonl,
    then map to a weight multiplier. If no scores file exists,
    default to 1.0.
    """
    if not REVIEWER_SCORES_PATH.exists():
        return 1.0

    raw_score = 0.0
    for entry in load_jsonl(REVIEWER_SCORES_PATH):
        if entry.get("reviewer_id") == reviewer_id:
            raw_score = float(entry.get("score", 0.0))
            break

    if raw_score >= 0.75:
        return 1.25
    elif raw_score >= 0.5:
        return 1.0
    else:
        return 0.75


def get_adaptive_threshold(reviewer_weight: float) -> float:
    """
    Return a suppression threshold based on the reviewer's weight tier:
     - weight >= 1.25 → threshold 0.4
     - 0.85 < weight < 1.25 → default 0.7
     - weight <= 0.85 → threshold 0.8
    """
    if reviewer_weight >= 1.25:
        return 0.4
    elif reviewer_weight <= 0.85:
        return 0.8
    else:
        return 0.7


# --- Endpoints ---------------------------------------

@router.post("/flag-for-retraining", status_code=200)
async def flag_for_retraining(req: RetrainRequest):
    """
    Records a signal for later retraining, including the reviewer's weight.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    weight = get_reviewer_weight(req.reviewer_id or "")

    entry = {
        "signal_id":       req.signal_id,
        "reviewer_id":     req.reviewer_id,
        "reason":          req.reason,
        "note":            req.note,
        "reviewer_weight": weight,
        "timestamp":       time.time(),
    }

    with RETRAINING_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    return {
        "status":          "queued",
        "signal_id":       req.signal_id,
        "reviewer_weight": weight,
    }


@router.post("/override-suppression", status_code=200)
async def override_suppression(req: OverrideRequest):
    """
    Applies a manual override to a suppressed signal,
    using reviewer_weight and adaptive threshold.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    weight = get_reviewer_weight(req.reviewer_id or "")
    old_score = 0.0  # assume read from signal store; tests mock this
    weighted_delta = req.trust_delta * weight
    new_score = old_score + weighted_delta

    threshold = get_adaptive_threshold(weight)
    unsuppressed = new_score >= threshold

    entry = {
        "signal_id":        req.signal_id,
        "reviewer_id":      req.reviewer_id,
        "action":           "override_suppression",
        "trust_delta":      req.trust_delta,
        "reviewer_weight":  weight,
        "weighted_delta":   weighted_delta,
        "old_score":        old_score,
        "new_score":        new_score,
        "threshold_used":   threshold,
        "unsuppressed":     unsuppressed,
        "timestamp":        time.time(),
    }

    with RETRAINING_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    return {
        "reviewer_weight": weight,
        "new_trust_score": new_score,
        "unsuppressed":    unsuppressed,
    }


# (Your existing /adjust-signals-based-on-feedback endpoint remains here)