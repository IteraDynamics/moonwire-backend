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

def generate_composite_signal(asset, source_scores):
    # Filter out any invalid scores
    valid_scores = [s for s in source_scores if isinstance(s.get("sentiment_score"), (int, float))]

    if not valid_scores:
        return None  # optionally raise error

    # Average sentiment
    avg_score = round(sum(s["sentiment_score"] for s in valid_scores) / len(valid_scores), 4)

    # Composite confidence
    confidences = [s.get("confidence", "low") for s in valid_scores]
    if "high" in confidences:
        confidence = "high"
    elif "medium" in confidences:
        confidence = "medium"
    else:
        confidence = "low"

    # Signal label
    if avg_score >= 0.6:
        label = "Bullish Momentum"
    elif avg_score <= -0.6:
        label = "Bearish Reversal"
    else:
        label = "Neutral Drift"

    # Signal trend
    if avg_score > 0:
        trend = "upward"
    elif avg_score < 0:
        trend = "downward"
    else:
        trend = "flat"

    # Breakdown
    source_breakdown = {s["source"]: s["sentiment_score"] for s in valid_scores}
    top_drivers = list(source_breakdown.keys())

    return {
        "id": f"sig_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.utcnow().isoformat(),
        "asset": asset,
        "score": avg_score,
        "confidence": confidence,
        "label": label,
        "trend": trend,
        "top_drivers": top_drivers,
        "fallback_type": "composite",
        "source_breakdown": source_breakdown
    }
