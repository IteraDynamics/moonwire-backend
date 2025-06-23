# src/adjustment_trigger_router.py

from fastapi import APIRouter
from src.adjust_signals_based_on_feedback import adjust_signals
from src.cache_instance import cache
from datetime import datetime

router = APIRouter()

@router.post("/internal/adjust-signals-based-on-feedback")
def trigger_adjust_signals():
    """
    Primary adjustment endpoint — processes historical signal log entries.
    """
    return adjust_signals()

@router.post("/internal/adjust-cache-signals")
def adjust_from_cache():
    """
    NEW: Adjusts in-memory cache signals based on disagreement model.
    Adds model_adjusted signals to history in place.
    """
    summary = []
    timestamp = datetime.utcnow().isoformat()

    for key in cache.keys():
        if not key.endswith("_history"):
            continue

        history = cache.get_signal(key)
        if not history or not isinstance(history, list):
            continue

        latest = history[-1]
        if latest.get("adjustment_applied"):
            continue

        payload = {
            "score": latest.get("score", 0.5),
            "confidence": latest.get("confidence", 0.5),
            "label": latest.get("label", "Neutral")
        }

        from src.feedback_prediction_router import predict_disagreement
        prediction = predict_disagreement(payload)

        if prediction.get("probability", 0) > 0.7:
            adjusted_signal = {
                **latest,
                "confidence": round(latest.get("confidence", 0.5) * 0.9, 4),
                "adjustment_applied": True,
                "adjustment_reason": "model_disagreement_risk",
                "adjusted_at": timestamp,
                "type": "model_adjusted"
            }
            updated = history + [adjusted_signal]
            cache.set_signal(key, updated)

            summary.append({
                "asset": key.replace("_history", ""),
                "status": "adjusted",
                "confidence_delta": round(latest.get("confidence", 0.5) - adjusted_signal["confidence"], 4)
            })
        else:
            summary.append({
                "asset": key.replace("_history", ""),
                "status": "ok",
                "probability": prediction.get("probability", 0)
            })

    return {"summary": summary}