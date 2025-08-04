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
RETRAIN_LOG_PATH = BASE_DIR / "logs" / "retraining_log.jsonl"

router = APIRouter(prefix="/internal")


def get_adaptive_threshold(reviewer_weight: float) -> float:
    """
    Return a suppression threshold based on reviewer weight.
    """
    if reviewer_weight >= 1.2:
        return 0.6
    elif reviewer_weight <= 0.85:
        return 0.8
    else:
        return 0.7


class RetrainRequest(BaseModel):
    signal_id: str
    reason: str
    reviewer_id: str
    note: Optional[str] = None


@router.post("/flag-for-retraining", status_code=200)
async def flag_for_retraining(req: RetrainRequest):
    """
    Records a signal for later retraining, storing reviewer_weight for prioritization.
    """
    # Ensure logs directory exists
    RETRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Determine weight
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

        if score >= 0.75:
            weight = 1.25
        elif score >= 0.5:
            weight = 1.0
        else:
            weight = 0.75

    # Assemble log entry
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
    signal_id: str
    override_reason: str
    reviewer_id: str
    trust_delta: float
    note: Optional[str] = None


@router.post("/override-suppression", status_code=200)
async def override_suppression(req: OverrideRequest):
    """
    Applies a manual override: weights trust_delta by reviewer_weight, unsuppress if new_score >= adaptive threshold.
    """
    # Determine weight
    if not REVIEWER_SCORES_PATH.exists():
        weight = 1.0
    else:
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
        if score >= 0.75:
            weight = 1.25
        elif score >= 0.5:
            weight = 1.0
        else:
            weight = 0.75

    # Calculate weighted delta and new score
    old_score = 0.0  # TODO: retrieve actual trust score
    weighted_delta = weight * req.trust_delta
    new_score = old_score + weighted_delta

    # Adaptive threshold
    threshold = get_adaptive_threshold(weight)
    unsuppressed = new_score >= threshold

    # Debug print
    print("🚨 /internal/override-suppression hit")
    print(f"  reviewer_id={req.reviewer_id}, reviewer_weight={weight}")
    print(f"  trust_delta={req.trust_delta}, weighted_delta={weighted_delta}")
    print(f"  old_score={old_score}, new_score={new_score}")
    print(f"  threshold_used={threshold}, unsuppressed={unsuppressed}")

    return {
        "override_applied": True,
        "signal_id": req.signal_id,
        "reviewer_weight": weight,
        "new_trust_score": new_score,
        "threshold_used": threshold,
        "unsuppressed": unsuppressed,
    }


@router.post("/adjust-signals-based-on-feedback", status_code=200)
async def trigger_adjust_signals():
    result = predict_disagreement()
    return {"adjustment_result": result}
