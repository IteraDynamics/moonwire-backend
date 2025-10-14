# src/feedback_analysis_router.py

from fastapi import APIRouter

router = APIRouter()

@router.get("/feedback/aggregate")
def get_feedback_stats():
    """
    Returns mock aggregated feedback data for assets.
    Enables model disagreement audit view in frontend.
    """
    return [
        {
            "asset": "BTC",
            "user_feedback": "Too bullish",
            "confidence": 0.85,
            "reliability_score": 0.91,
            "timestamp": "2025-06-08T18:00:00Z"
        },
        {
            "asset": "ETH",
            "user_feedback": "Neutral is fair",
            "confidence": 0.6,
            "reliability_score": 0.64,
            "timestamp": "2025-06-08T19:00:00Z"
        },
        {
            "asset": "BTC",
            "user_feedback": "Not aligned with volume",
            "confidence": 0.3,
            "reliability_score": 0.37,
            "timestamp": "2025-06-08T20:00:00Z"
        }
    ]