import uuid
from datetime import datetime

def generate_signal(
    asset: str,
    sentiment_score: float,
    fallback_type: str,
    top_drivers: list
) -> dict:
    # Normalize score
    score = round(sentiment_score, 2)

    # Confidence logic
    if score >= 0.7:
        confidence = "high"
    elif score >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    # Trend logic
    trend = "upward" if score >= 0.5 else "downward"

    # Labeling logic
    if score >= 0.75:
        label = "Bullish Momentum Spike"
    elif score <= 0.25:
        label = "Bearish Reversal"
    else:
        label = "Neutral Drift"

    return {
        "id": f"sig_{uuid.uuid4().hex[:8]}",
        "asset": asset,
        "score": score,
        "confidence": confidence,
        "label": label,
        "trend": trend,
        "top_drivers": top_drivers,
        "timestamp": datetime.utcnow().isoformat(),
        "fallback_type": fallback_type
    }
