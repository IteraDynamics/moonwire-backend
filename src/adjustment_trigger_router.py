# src/adjustment_trigger_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import json
from datetime import datetime

from scripts.ml_utils.train_feedback_disagreement_model import predict_disagreement

# ——— Module‐level router export ———
router = APIRouter(prefix="/internal")

# ——— Log file paths ———
BASE_DIR               = Path(__file__).resolve().parent.parent
REVIEWER_SCORES_PATH   = BASE_DIR / "logs" / "reviewer_scores.jsonl"
RETRAIN_LOG_PATH       = BASE_DIR / "logs" / "retraining_log.jsonl"


def get_adaptive_threshold(reviewer_weight: float) -> float:
    """
    Return a suppression threshold based on reviewer weight.
    High-trust reviewers (>=1.25) unsuppress easier at 0.4
    Mid-trust reviewers use default 0.7
    Low-trust reviewers (<=0.85) need stronger evidence at 0.8
    """
    if reviewer_weight >= 1.25:
        return 0.4
    elif reviewer_weight <= 0.85:
        return 0.8
    else:
        return 0.7


class RetrainRequest(BaseModel):
    signal_id:   str
    reason:      str
    reviewer_id: str
    note:        Optional[str] = None


@router.post("/flag-for-retraining", status_code=200)
async def flag_for_retraining(req: RetrainRequest):
    """
    Records retrain flags with reviewer_weight for prioritization.
    """
    RETRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Lookup weight
    if not REVIEWER_SCORES_PATH.exists():
        weight = 1.0
    else:
        score = 0.0
        try:
            with REVIEWER_SCORES_PATH.open() as f:
                for line in f:
                    e = json.loads(line)
                    if e.get("reviewer_id") == req.reviewer_id:
                        score = e.get("score", 0.0)
                        break
        except Exception as e:
            raise HTTPException(500, f"Error reading scores: {e}")

        if score >= 0.75:
            weight = 1.25
        elif score >= 0.5:
            weight = 1.0
        else:
            weight = 0.75

    entry = {
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "signal_id":       req.signal_id,
        "reviewer_id":     req.reviewer_id,
        "reason":          req.reason,
        "note":            req.note,
        "reviewer_weight": weight,
    }
    try:
        with RETRAIN_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        raise HTTPException(500, f"Error writing log: {e}")

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
    Applies a manual override, using an adaptive threshold per reviewer tier.
    """
    # Lookup weight
    if not REVIEWER_SCORES_PATH.exists():
        weight = 1.0
    else:
        score = 0.0
        try:
            with REVIEWER_SCORES_PATH.open() as f:
                for line in f:
                    e = json.loads(line)
                    if e.get("reviewer_id") == req.reviewer_id:
                        score = e.get("score", 0.0)
                        break
        except Exception as e:
            raise HTTPException(500, f"Error reading scores: {e}")

        if score >= 0.75:
            weight = 1.25
        elif score >= 0.5:
            weight = 1.0
        else:
            weight = 0.75

    old_score = 0.0  # placeholder for existing trust
    weighted_delta = weight * req.trust_delta
    new_score = old_score + weighted_delta

    threshold = get_adaptive_threshold(weight)
    unsuppressed = new_score >= threshold

    print("🚨 /internal/override-suppression hit")
    print(f"  reviewer_id={req.reviewer_id}, reviewer_weight={weight}")
    print(f"  trust_delta={req.trust_delta}, weighted_delta={weighted_delta}")
    print(f"  old_score={old_score}, new_score={new_score}")
    print(f"  threshold_used={threshold}, unsuppressed={unsuppressed}")

    return {
        "override_applied": True,
        "signal_id":        req.signal_id,
        "reviewer_weight":  weight,
        "new_trust_score":  new_score,
        "threshold_used":   threshold,
        "unsuppressed":     unsuppressed,
    }


@router.post("/adjust-signals-based-on-feedback", status_code=200)
async def trigger_adjust_signals():
    result = predict_disagreement()
    return {"adjustment_result": result}
