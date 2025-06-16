# src/threshold_simulator.py

import json
from collections import defaultdict

# === Mock combined signal + feedback history ===
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

def simulate_thresholds(data, min_confidence=0.8):
    result = defaultdict(lambda: {
        "total_feedback": 0,
        "adjusted_signals": 0,
        "avg_confidence": 0.0,
        "high_confidence_disagreements": 0
    })

    for entry in data:
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

    # Finalize average confidence
    for asset in result:
        tf = result[asset]["total_feedback"]
        result[asset]["avg_confidence"] = round(result[asset]["avg_confidence"] / tf, 3)

    return result

if __name__ == "__main__":
    output = simulate_thresholds(mock_entries, min_confidence=0.8)
    print(json.dumps(output, indent=2))