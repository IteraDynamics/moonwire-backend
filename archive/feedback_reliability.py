# src/feedback_reliability.py

import json
from datetime import datetime, timedelta

# === MOCK FEEDBACK DATA ===
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

# === RELIABILITY SCORING ===
def compute_reliability(feedback_entries):
    results = []

    for entry in feedback_entries:
        if entry.get("type") != "user_feedback":
            continue

        confidence = entry.get("confidence", 0)
        reliability_score = round(confidence * 0.9 + 0.1, 3)  # Basic confidence-weighted stub

        results.append({
            "asset": entry.get("asset"),
            "user_feedback": entry.get("user_feedback"),
            "confidence": confidence,
            "reliability_score": reliability_score,
            "timestamp": entry.get("timestamp")
        })

    return results

if __name__ == "__main__":
    reliability_results = compute_reliability(mock_feedback_data)
    print(json.dumps(reliability_results, indent=2))
