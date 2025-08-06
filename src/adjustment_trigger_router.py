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

class RetrainRequest(BaseModel):
    signal_id: str
    reason:    str
    note:      Optional[str] = None
    reviewer_id: Optional[str] = None  # allow upstream to pass reviewer_id if desired

def load_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]

def get_reviewer_weight(reviewer_id: str) -> float:
    """Lookup a reviewer's score in reviewer_scores.jsonl, default 1.0."""
    if REVIEWER_SCORES_PATH.exists():
        for entry in load_jsonl(REVIEWER_SCORES_PATH):
            if entry.get("reviewer_id") == reviewer_id:
                # assume score field holds the numeric score
                return float(entry.get("score", 1.0))
    return 1.0

@router.post("/flag-for-retraining", status_code=200)
async def flag_for_retraining(req: RetrainRequest):
    """
    Records a signal for later retraining, including the reviewer's weight.
    """
    # 1) Determine reviewer_id (explicit or default to 'unknown')
    reviewer = req.reviewer_id or "unknown"

    # 2) Lookup weight (fallback to 1.0 if no scores file or no entry)
    weight = get_reviewer_weight(reviewer)

    # 3) Ensure logs directory and file exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    RETRAINING_LOG_PATH.touch(exist_ok=True)

    # 4) Build log entry
    entry = {
        "timestamp":       time.time(),
        "signal_id":       req.signal_id,
        "reviewer_id":     reviewer,
        "reason":          req.reason,
        "note":            req.note,
        "reviewer_weight": weight,
    }

    # 5) Append to retraining_log.jsonl
    try:
        with RETRAINING_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write retraining log: {e}")

    # 6) Return success payload matching tests
    return {
        "status":          "queued",
        "signal_id":       req.signal_id,
        "reviewer_weight": weight,
    }

class OverrideRequest(BaseModel):
    signal_id:        str
    override_reason:  str
    note:             Optional[str] = None
    reviewer_id:      Optional[str] = None
    trust_delta:      Optional[float] = None

@router.post("/override-suppression", status_code=200)
async def override_suppression(req: OverrideRequest):
    """
    Applies a manual override to a suppressed signal, identical pattern.
    """
    # (Existing logic…) 
    # You can copy your current override logic here, but ensure you:
    # - os.makedirs(LOGS_DIR, exist_ok=True)
    # - RETRAINING_LOG_PATH.touch(exist_ok=True) if you log elsewhere
    # - Return the expected JSON per tests.
    raise HTTPException(status_code=501, detail="Override logic not shown here")