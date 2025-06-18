# src/model_disagreement_router.py

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# 🔧 Input schema to match frontend call
class SignalInput(BaseModel):
    score: float
    confidence: float
    label: str
    fallback_type: str

@router.post("/internal/predict-feedback-risk")
def predict_feedback_disagreement(input: SignalInput):
    """
    Simulates disagreement prediction for signal inputs using a placeholder model.
    Currently returns mock logic until real training is in place.
    """
    # Simple heuristic fallback model for now
    # High score + low confidence = risky
    # Low score + high confidence = risky
    # Mid score + mid confidence = stable
    s = input.score
    c = input.confidence

    # Basic disagreement risk logic
    disagreement_prob = abs(s - 0.5) * (1 - c)
    disagree = disagreement_prob > 0.5

    return {
        "likely_disagreed": disagree,
        "probability": round(disagreement_prob, 2)
    }