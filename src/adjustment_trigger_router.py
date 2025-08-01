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
    signal_id:  str
    reason:     str
    reviewer_id:str
    note:       Optional[str] = None


@router.post("/flag-for-retraining", status_code=200)
async def flag_for_retraining(req: RetrainRequest):
    """
    Records a signal for later retraining, storing reviewer_weight for prioritization.
    """
    # Ensure logs directory exists
    RETRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If no scores file, default weight to 1.0
    if not REVIEWER_SCORES_PATH.exists():
        weight = 1.0
    else:
        # Lookup reviewer score
        score = 0.0
        try:
            with REVIEWER_SCORES_PATH.open("r") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("reviewer_id") == req.reviewer_id:
                        score = entry.get("score", 0.0)
                        break
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading reviewer scores: {e}")

        # Compute weight based on existing score
        if score >= 0.75:
            weight = 1.25
        elif score >= 0.5:
            weight = 1.0
        else:
            weight = 0.75

    # Assemble retrain log entry
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "signal_id": req.signal_id,
        "reviewer_id": req.reviewer_id,
        "reason": req.reason,
        "note": req.note,
        "reviewer_weight": weight,
    }
    try:
        with RETRAIN_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing retraining log: {e}")

    # Debug print
    print(f"🚨 /internal/flag-for-retraining hit -> reviewer_weight={weight}")

    return {"status": "queued", "reviewer_weight": weight}


class OverrideRequest(BaseModel):
    signal_id:       str
    override_reason: str
    reviewer_id:     str
    trust_delta:     float
    note:            Optional[str] = None


@router.post("/override-suppression", status_code=200)
async def override_suppression(req: OverrideRequest):
    """
    Applies a manual override, weighting trust_delta by reviewer_weight,
    auto-unsuppressing if new_trust_score >= threshold.
    """
    # If no scores file, default weight to 1.0
    if not REVIEWER_SCORES_PATH.exists():
        weight = 1.0
    else:
        # Lookup reviewer score
        score = 0.0
        try:
            with REVIEWER_SCORES_PATH.open("r") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("reviewer_id") == req.reviewer_id:
                        score = entry.get("score", 0.0)
                        break
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading reviewer scores: {e}")

        # Compute weight based on existing score
        if score >= 0.75:
            weight = 1.25
        elif score >= 0.5:
            weight = 1.0
        else:
            weight = 0.75

    # Calculate new trust score
    old_score = 0.0  # TODO: retrieve actual existing trust score
    new_score = old_score + weight * req.trust_delta
    threshold = 0.4
    unsuppressed = new_score >= threshold

    # Debug print
    print(f"🚨 /internal/override-suppression hit")
    print(f"  reviewer_id={req.reviewer_id}, reviewer_weight={weight}")
    print(f"  old_score={old_score}, trust_delta={req.trust_delta}, new_score={new_score}, unsuppressed={unsuppressed}")

    return {
        "override_applied": True,
        "signal_id": req.signal_id,
        "reviewer_weight": weight,
        "new_trust_score": new_score,
        "unsuppressed": unsuppressed,
    }


@router.post("/adjust-signals-based-on-feedback", status_code=200)
async def trigger_adjust_signals():
    result = predict_disagreement()
    return {"adjustment_result": result}
