# src/feedback_analysis_router.py

from fastapi import APIRouter
from datetime import datetime
from typing import List

router = APIRouter(prefix="/feedback", tags=["feedback-analysis"])

# === Mock feedback input ===
mock_feedback_data = [
    {
        "type": "user_feedback",
        "timestamp": "2025-06-08T18:00:00Z",
        "asset": "BTC",
        "sentiment": 0.45,
        "user_feedback": "Too bullish",
        "confidence": 0.9,
        "source": "frontend"
    },
    {
        "type": "user_feedback",
        "timestamp": "2025-06-08T19:00:00Z",
        "asset": "ETH",
        "sentiment": 0.35,
        "user_feedback": "Fair",
        "confidence": 0.6,
        "source": "frontend"
    },
    {
        "type": "user_feedback",
        "timestamp": "2025-06-08T20:00:00Z",
        "asset": "BTC",
        "sentiment": 0.42,
        "user_feedback": "Not aligned with volume",
        "confidence": 0.3,
        "source": "frontend"
    }
]

# === Reliability scoring logic ===
def compute_reliability(feedback_entries):
    results = []
    for entry in feedback_entries:
        confidence = entry.get("confidence", 0)
        reliability_score = round(confidence * 0.9 + 0.1, 3)  # Placeholder formula
        results.append({
            "asset": entry.get("asset"),
            "user_feedback": entry.get("user_feedback"),
            "confidence": confidence,
            "reliability_score": reliability_score,
            "timestamp": entry.get("timestamp")
        })
    return results

@router.get("/reliability")
def get_mock_feedback_reliability():
    return compute_reliability(mock_feedback_data)