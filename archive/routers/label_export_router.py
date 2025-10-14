# src/label_export_router.py

from fastapi import APIRouter

router = APIRouter(prefix="/feedback", tags=["feedback-export"])

# === Mock feedback data (same as label_export.py) ===
mock_feedback = [
    {
        "type": "user_feedback",
        "timestamp": "2025-06-08T18:00:00Z",
        "asset": "BTC",
        "sentiment": 0.45,
        "user_feedback": "Too bullish",
        "confidence": 0.9,
        "context": "Disagrees with price trend"
    },
    {
        "type": "user_feedback",
        "timestamp": "2025-06-08T19:00:00Z",
        "asset": "ETH",
        "sentiment": 0.35,
        "user_feedback": "Neutral is fair",
        "confidence": 0.6,
        "context": "Aligned with recent news"
    },
    {
        "type": "user_feedback",
        "timestamp": "2025-06-08T20:00:00Z",
        "asset": "BTC",
        "sentiment": 0.42,
        "user_feedback": "Too optimistic",
        "confidence": 0.3,
        "context": "Volume dropping, sentiment rising"
    }
]

def compute_reliability(confidence: float) -> float:
    return round(confidence * 0.9 + 0.1, 3)

@router.get("/export-labels")
def get_label_export_data():
    output = []

    for entry in mock_feedback:
        if entry.get("type") != "user_feedback":
            continue

        output.append({
            "asset": entry["asset"],
            "timestamp": entry["timestamp"],
            "user_feedback": entry["user_feedback"],
            "confidence": entry["confidence"],
            "reliability_score": compute_reliability(entry["confidence"]),
            "context": entry.get("context", "")
        })

    return output
