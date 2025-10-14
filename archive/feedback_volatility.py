# src/feedback_volatility.py

import json
import statistics
from collections import defaultdict

# === Mock feedback data ===
mock_feedback = [
    {
        "type": "user_feedback",
        "asset": "BTC",
        "confidence": 0.9,
        "agrees_with_signal": False
    },
    {
        "type": "user_feedback",
        "asset": "BTC",
        "confidence": 0.6,
        "agrees_with_signal": True
    },
    {
        "type": "user_feedback",
        "asset": "BTC",
        "confidence": 0.5,
        "agrees_with_signal": False
    },
    {
        "type": "user_feedback",
        "asset": "ETH",
        "confidence": 0.8,
        "agrees_with_signal": True
    },
    {
        "type": "user_feedback",
        "asset": "ETH",
        "confidence": 0.4,
        "agrees_with_signal": False
    }
]

def analyze_volatility(feedback_data):
    result = {}

    grouped = defaultdict(list)
    for entry in feedback_data:
        if entry.get("type") != "user_feedback":
            continue
        grouped[entry["asset"]].append(entry)

    for asset, entries in grouped.items():
        confidences = [e["confidence"] for e in entries]
        disagreements = [e for e in entries if not e.get("agrees_with_signal", True)]

        result[asset] = {
            "feedback_count": len(entries),
            "disagreement_rate": round(len(disagreements) / len(entries), 2),
            "mean_confidence": round(sum(confidences) / len(confidences), 3),
            "confidence_std_dev": round(statistics.stdev(confidences), 3) if len(confidences) > 1 else 0
        }

    return result

if __name__ == "__main__":
    out = analyze_volatility(mock_feedback)
    print(json.dumps(out, indent=2))