# src/adjustment_trigger_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import json
from datetime import datetime

from scripts.ml_utils.train_feedback_disagreement_model import predict_disagreement

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
REVIEWER_SCORES_PATH = BASE_DIR / "logs" / "reviewer_scores.jsonl"
RETRAIN_LOG_PATH      = BASE_DIR / "logs" / "retraining_log.jsonl"

router = APIRouter(prefix="/internal")


class RetrainRequest(BaseModel):
    signal_id: str
    reason:    str
    reviewer_id: str
    note:      Optional[str] = None


@router.post("/flag-for-retraining", status_code=200)
async def flag_for_retraining(req: RetrainRequest):
    """
    Records a signal for later retraining, 
    storing the reviewer_weight for downstream prioritization.
    """
    # ensure logs directory exists
    RETRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # load reviewer score if available
    score = None
    if REVIEWER_SCORES_PATH.exists():
        try:
            with REVIEWER_SCORES_PATH.open("r") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("reviewer_id") == req.reviewer_id:
                        score = entry.get("score", 0.0)
                        break
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading reviewer scores: {e}")

    # compute weight: default 1.0 if no score entry found
    if score is None:
        weight = 1.0
    else:
        if score >= 0.75:
            weight = 1.25
        elif score >= 0.5:
            weight = 1.0
        else:
            weight = 0.75

    # assemble log entry
    entry = {
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "signal_id":       req.signal_id,
        "reviewer_id":     req.reviewer_id,
        "reason":          req.reason,
        "note":            req.note,
        "reviewer_weight": weight,
    }

    # append to JSONL
    try:
        with RETRAIN_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing retraining log: {e}")

    # debug print
    print(f"🚨 /internal/flag-for-retraining hit")
    print(f"  signal_id: {req.signal_id}, reviewer_id: {req.reviewer_id}, weight: {weight}")
    print(f"  reason: {req.reason}, note: {req.note}")

    return {"status": "queued", "reviewer_weight": weight}


@router.post("/override-suppression", status_code=200)
async def override_suppression(req: BaseModel):
    # existing override logic…
    return {"override_applied": True, "signal_id": getattr(req, "signal_id", None)}


@router.post("/adjust-signals-based-on-feedback", status_code=200)
async def trigger_adjust_signals():
    result = predict_disagreement()
    return {"adjustment_result": result}