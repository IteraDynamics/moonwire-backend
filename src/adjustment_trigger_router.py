# src/adjustment_trigger_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import time
import json
import os

from src.paths import LOGS_DIR, RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH

router = APIRouter(prefix="/internal")


def get_adaptive_threshold(reviewer_weight: float) -> float:
    """
    Returns an adaptive suppression threshold based on reviewer_weight.
    """
    if reviewer_weight >= 1.25:
        return 0.4   # high‐trust reviewers unsuppress easier
    elif reviewer_weight <= 0.85:
        return 0.8   # low‐trust reviewers need stronger justification
    else:
        return 0.7   # default


class RetrainRequest(BaseModel):
    signal_id:   str
    reason:      str
    note:        Optional[str] = None
    reviewer_id: Optional[str] = None


@router.post("/flag-for-retraining", status_code=200)
async def flag_for_retraining(req: RetrainRequest):
    reviewer = req.reviewer_id or "unknown"
    weight = get_reviewer_weight(reviewer)

    os.makedirs(LOGS_DIR, exist_ok=True)
    RETRAINING_LOG_PATH.touch(exist_ok=True)

    entry = {
        "timestamp":       time.time(),
        "signal_id":       req.signal_id,
        "reviewer_id":     reviewer,
        "reason":          req.reason,
        "note":            req.note,
        "reviewer_weight": weight,
    }

    try:
        with RETRAINING_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "queued", "signal_id": req.signal_id, "reviewer_weight": weight}


class OverrideRequest(BaseModel):
    signal_id:        str
    override_reason:  str
    note:             Optional[str] = None
    reviewer_id:      Optional[str] = None
    trust_delta:      float


@router.post("/override-suppression", status_code=200)
async def override_suppression(req: OverrideRequest):
    reviewer = req.reviewer_id or "unknown"
    weight = get_reviewer_weight(reviewer)

    weighted_delta = req.trust_delta * weight
    old_score = 0.0
    new_score = old_score + weighted_delta

    threshold = get_adaptive_threshold(weight)
    unsuppressed = new_score >= threshold

    os.makedirs(LOGS_DIR, exist_ok=True)
    # (your existing JSONL append for override log here)

    return {
        "signal_id":       req.signal_id,
        "reviewer_weight": weight,
        "new_trust_score": new_score,
        "unsuppressed":    unsuppressed,
    }


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def get_reviewer_weight(reviewer_id: str) -> float:
    raw = 0.0
    if REVIEWER_SCORES_PATH.exists():
        for e in load_jsonl(REVIEWER_SCORES_PATH):
            if e.get("reviewer_id") == reviewer_id:
                raw = float(e.get("score", 0.0))
                break

    if raw >= 0.75:
        return 1.25
    elif raw >= 0.5:
        return 1.0
    else:
        return 0.75