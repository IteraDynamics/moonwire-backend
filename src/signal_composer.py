from datetime import datetime
import uuid

def generate_signal(asset, sentiment_score, price_at_score=None, fallback_type="mock"):
    """
    Generate a structured signal object from sentiment and market context.
    This function is rules-based for now and can be swapped with model logic later.
    """
    # Rule-based logic
    if sentiment_score > 0.6:
        label = "Bullish Momentum Spike"
        confidence = "high"
        trend = "upward"
    elif sentiment_score < -0.4:
        label = "Bearish Sentiment Breakdown"
        confidence = "medium"
        trend = "downward"
    else:
        label = "Neutral Market Noise"
        confidence = "low"
        trend = "flat"

    return {
        "id": f"sig_{uuid.uuid4().hex[:8]}",
        "asset": asset,
        "score": round(sentiment_score, 4),
        "confidence": confidence,
        "label": label,
        "trend": trend,
        "top_drivers": ["social sentiment", "volume shift"],  # Placeholder
        "timestamp": datetime.utcnow().isoformat(),
        "fallback_type": fallback_type
    }