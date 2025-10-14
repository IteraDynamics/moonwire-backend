# src/threshold_simulator_router.py

from fastapi import APIRouter, Query
from collections import defaultdict

router = APIRouter(prefix="/internal", tags=["internal-tools"])

# === Same mock data used in CLI module ===
mock_entries = [
    {
        "type": "user_feedback",
        "asset": "BTC",
        "timestamp": "2025-06-10T14:00:00Z",
        "confidence": 0.91,
        "user_feedback": "Too bullish",
        "agrees_with_signal": False
    },
    {
        "type": "user_feedback",
        "asset": "BTC",
        "timestamp": "2025-06-10T15:00:00Z",
        "confidence": 0.45,
        "user_feedback": "Seems fine",
        "agrees_with_signal": True
    },
    {
        "type": "user_feedback",
        "asset": "ETH",
        "timestamp": "2025-06-10T14:30:00Z",
        "confidence": 0.85,
        "user_feedback": "Underestimated",
        "agrees_with_signal": False
    },
    {
        "type": "user_feedback",
        "asset": "ETH",
        "timestamp": "2025-06-10T16:30:00Z",
        "confidence": 0.6,
        "user_feedback": "Neutral",
        "agrees_with_signal": True
    }
]

@router.get("/simulate-thresholds")
def simulate_feedback_thresholds(min_confidence: float = Query(0.8)):
    result = defaultdict(lambda: {
        "total_feedback": 0,
        "adjusted_signals": 0,
        "avg_confidence": 0.0,
        "high_confidence_disagreements": 0
    })

    for entry in mock_entries:
        if entry.get("type") != "user_feedback":
            continue

        asset = entry["asset"]
        confidence = entry["confidence"]
        agrees = entry["agrees_with_signal"]

        result[asset]["total_feedback"] += 1
        result[asset]["avg_confidence"] += confidence

        if confidence >= min_confidence and not agrees:
            result[asset]["adjusted_signals"] += 1
            result[asset]["high_confidence_disagreements"] += 1

    for asset in result:
        tf = result[asset]["total_feedback"]
        result[asset]["avg_confidence"] = round(result[asset]["avg_confidence"] / tf, 3)

    return result