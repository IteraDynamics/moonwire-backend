# src/adjustment_trigger_router.py

from fastapi import APIRouter
from scripts.ml_utils.train_feedback_disagreement_model import predict_disagreement  # ✅ Updated import

router = APIRouter()

@router.post("/internal/adjust-signals-based-on-feedback")
def trigger_adjust_signals():
    return predict_disagreement()