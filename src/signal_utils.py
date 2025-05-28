# src/signal_utils.py

from datetime import datetime
import uuid

def generate_signal(asset, score, source, fallback_type=None, top_drivers=None, timestamp=None):
    return {
        "id": f"sig_{uuid.uuid4().hex[:8]}",
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "asset": asset,
        "score": score,
        "confidence": "medium" if abs(score) >= 0.4 else "low",
        "label": "Bullish Momentum" if score >= 0.6 else "Bearish Reversal" if score <= -0.6 else "Neutral Drift",
        "trend": "upward" if score > 0 else "downward" if score < 0 else "flat",
        "top_drivers": top_drivers or [],
        "fallback_type": fallback_type,
    }
