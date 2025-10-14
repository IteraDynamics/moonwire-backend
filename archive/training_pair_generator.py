# src/training_pair_generator.py

import json
from datetime import datetime, timedelta

# === Mock signal + feedback log (usually from signal_history.jsonl) ===
mock_history = [
    {
        "type": "signal",
        "timestamp": "2025-06-16T14:00:00Z",
        "asset": "BTC",
        "score": 0.42,
        "confidence": 0.85,
        "label": "Positive",
        "fallback_type": "mock"
    },
    {
        "type": "user_feedback",
        "timestamp": "2025-06-16T14:10:00Z",
        "asset": "BTC",
        "user_feedback": "Too bearish",
        "confidence": 0.9,
        "reliability_score": 0.85,
        "agrees_with_signal": False
    },
    {
        "type": "user_feedback",
        "timestamp": "2025-06-16T14:45:00Z",
        "asset": "ETH",
        "user_feedback": "Accurate",
        "confidence": 0.6,
        "reliability_score": 0.6,
        "agrees_with_signal": True
    },
    {
        "type": "signal",
        "timestamp": "2025-06-16T14:30:00Z",
        "asset": "ETH",
        "score": 0.38,
        "confidence": 0.72,
        "label": "Neutral",
        "fallback_type": "mock"
    }
]

def parse_iso(ts):
    return datetime.fromisoformat(ts.replace("Z", ""))

def generate_training_pairs(log_entries, window_minutes=30):
    pairs = []
    signals = [e for e in log_entries if e["type"] == "signal"]
    feedbacks = [e for e in log_entries if e["type"] == "user_feedback"]

    for signal in signals:
        t_sig = parse_iso(signal["timestamp"])
        asset = signal["asset"]

        for fb in feedbacks:
            if fb["asset"] != asset:
                continue

            t_fb = parse_iso(fb["timestamp"])
            if abs((t_fb - t_sig).total_seconds()) > window_minutes * 60:
                continue

            if not fb.get("agrees_with_signal", True):
                pairs.append({
                    "asset": asset,
                    "timestamp": signal["timestamp"],
                    "X": {
                        "score": signal["score"],
                        "confidence": signal["confidence"],
                        "label": signal["label"],
                        "fallback_type": signal.get("fallback_type", "")
                    },
                    "y": fb["user_feedback"],
                    "weight": round(fb["confidence"] * fb.get("reliability_score", 1.0), 3)
                })

    return pairs

if __name__ == "__main__":
    training_data = generate_training_pairs(mock_history)
    print(json.dumps(training_data, indent=2))
