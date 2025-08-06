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
    # 1) No scores yet → fallback
    if not REVIEWER_SCORES_PATH.exists():
        return 1.0

    # 2) Find their raw score
    raw = 0.0
    for entry in load_jsonl(REVIEWER_SCORES_PATH):
        if entry.get("reviewer_id") == reviewer_id:
            raw = float(entry.get("score", 0.0))
            break

    # 3) Map to weight
    if raw >= 0.75:
        return 1.25
    elif raw >= 0.5:
        return 1.0
    else:
        return 0.75

# --- Endpoints ---------------------------------------

@router.post("/flag-for-retraining", status_code=200)
async def flag_for_retraining(req: RetrainRequest):
    """
    Records a signal for later retraining, including the reviewer's weight.
    """
    # ensure logs directory
    os.makedirs(LOGS_DIR, exist_ok=True)

    # determine weight (defaults to 1.0 if no scores file / no entry)
    weight = get_reviewer_weight(req.reviewer_id or "")

    # build the log entry
    entry = {
        "signal_id":       req.signal_id,
        "reviewer_id":     req.reviewer_id,
        "reason":          req.reason,
        "note":            req.note,
        "reviewer_weight": weight,
        "timestamp":       time.time(),
    }

    # append to JSONL
    with RETRAINING_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    # return queued status
    return {
        "status":          "queued",
        "signal_id":       req.signal_id,
        "reviewer_weight": weight,
    }

# (existing override and adjust-signals endpoints would follow here)