from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import json
from datetime import datetime

from scripts.ml_utils.train_feedback_disagreement_model import predict_disagreement

# Paths
BASE_DIR               = Path(__file__).resolve().parent.parent
REVIEWER_SCORES_PATH   = BASE_DIR / "logs" / "reviewer_scores.jsonl"
RETRAIN_LOG_PATH       = BASE_DIR / "logs" / "retraining_log.jsonl"

router = APIRouter(prefix="/internal")


def get_adaptive_threshold(reviewer_weight: float) -> float:
    """
    Return a suppression threshold based on reviewer weight.
    High-trust reviewers can unsuppress easier (lower threshold),
    low-trust reviewers need stronger evidence (higher threshold).
    """
    if reviewer_weight >= 1.2:
        return 0.6
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
    # … unchanged from prior implementation …
    # (omitted here for brevity)
    ...


class OverrideRequest(BaseModel):
    signal_id:       str
    override_reason: str
    reviewer_id:     str
    trust_delta:     float
    note:            Optional[str] = None


@router.post("/override-suppression", status_code=200)
async def override_suppression(req: OverrideRequest):
    """
    Applies a manual override: looks up reviewer_weight, computes weighted trust_delta,
    uses an adaptive suppression threshold per reviewer tier, and logs the decision.
    """
    # Load reviewer score
    score = 0.0
    if REVIEWER_SCORES_PATH.exists():
        try:
            with REVIEWER_SCORES_PATH.open("r") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("reviewer_id") == req.reviewer_id:
                        score = entry.get("score", 0.0)
                        break
        except Exception as e:
            raise HTTPException(500, f"Failed reading reviewer scores: {e}")

    # Compute weight
    if score >= 0.75:
        weight = 1.25
    elif score >= 0.5:
        weight = 1.0
    else:
        weight = 0.75

    # Compute new trust and adaptive threshold
    old_score = 0.0  # TODO: load actual existing trust
    weighted_delta = weight * req.trust_delta
    new_score = old_score + weighted_delta

    threshold = get_adaptive_threshold(weight)
    unsuppressed = new_score >= threshold

    # Debug logging
    print("🚨 /internal/override-suppression hit")
    print(f"  reviewer_id={req.reviewer_id}, score={score}, weight={weight}")
    print(f"  trust_delta={req.trust_delta}, weighted_delta={weighted_delta}")
    print(f"  old_score={old_score}, new_score={new_score}")
    print(f"  adaptive_threshold={threshold}, unsuppressed={unsuppressed}")

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
