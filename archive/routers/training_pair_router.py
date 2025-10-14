# src/training_pair_router.py

from fastapi import APIRouter
from datetime import datetime, timedelta

router = APIRouter(prefix="/internal", tags=["internal-tools"])

# === Mock signal + feedback entries ===
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

@router.get("/generate-training-pairs")
def generate_training_pairs(min_conf: float = 0.0, window_minutes: int = 30):
    signals = [e for e in mock_history if e["type"] == "signal"]
    feedbacks = [e for e in mock_history if e["type"] == "user_feedback"]

    pairs = []

    for sig in signals:
        t_sig = parse_iso(sig["timestamp"])
        asset = sig["asset"]

        for fb in feedbacks:
            if fb["asset"] != asset:
                continue
            t_fb = parse_iso(fb["timestamp"])
            if abs((t_fb - t_sig).total_seconds()) > window_minutes * 60:
                continue
            if not fb.get("agrees_with_signal", True) and fb["confidence"] >= min_conf:
                pairs.append({
                    "asset": asset,
                    "timestamp": sig["timestamp"],
                    "X": {
                        "score": sig["score"],
                        "confidence": sig["confidence"],
                        "label": sig["label"],
                        "fallback_type": sig.get("fallback_type", "")
                    },
                    "y": fb["user_feedback"],
                    "weight": round(fb["confidence"] * fb.get("reliability_score", 1.0), 3)
                })

    return pairs
