# src/adjustment_trigger_router.py

from fastapi import APIRouter
from scripts.predict_disagreement import predict_disagreement

router = APIRouter()

@router.post("/internal/adjust-signals-based-on-feedback")
def trigger_adjust_signals():
    # ✅ Load data and run disagreement prediction
    result = predict_disagreement()

    # ✅ Return structured response for logging/debug
    return {
        "message": "Adjustment process complete",
        "adjusted_signals_count": result.get("adjusted_signals_count", 0),
        "details": result.get("details", [])
    }