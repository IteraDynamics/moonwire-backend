from fastapi import APIRouter
import requests
from datetime import datetime
from src.signal_log import log_signal
from src.cache_instance import cache

router = APIRouter()

# 🔧 Model API to predict disagreement risk
DISAGREEMENT_MODEL_URL = "https://moonwire-signal-engine-1.onrender.com/internal/predict-feedback-risk"

# 🔧 Score/Confidence adjustment cap
ADJUSTMENT_FACTOR = 0.9
RISK_THRESHOLD = 0.7

@router.post("/internal/adjust-signals-based-on-feedback")
def adjust_signals_from_feedback():
    results = []
    timestamp = datetime.utcnow().isoformat()

    for key in cache.keys():
        if not key.endswith("_history"):
            history = cache.get_signal(f"{key}_history")
            if not history or not isinstance(history, list):
                continue

            latest = history[-1]
            if latest.get("adjustment_applied"):
                continue  # Skip already-adjusted signals

            score = latest.get("score")
            confidence = latest.get("confidence", 0.5)
            label = latest.get("label", "Neutral")
            fallback_type = latest.get("fallback_type", "mock")

            payload = {
                "score": score,
                "confidence": confidence,
                "label": label,
                "fallback_type": fallback_type
            }

            try:
                model_resp = requests.post(DISAGREEMENT_MODEL_URL, json=payload)
                model_data = model_resp.json()
                disagreement_prob = model_data.get("probability", 0)

                if disagreement_prob > RISK_THRESHOLD:
                    # Soft-adjust score and/or confidence
                    adjusted_score = round(score * ADJUSTMENT_FACTOR, 4) if score is not None else None
                    adjusted_confidence = round(confidence * ADJUSTMENT_FACTOR, 4) if confidence is not None else None

                    adjusted_signal = {
                        **latest,
                        "score": adjusted_score,
                        "confidence": adjusted_confidence,
                        "timestamp": timestamp,
                        "adjustment_applied": True,
                        "adjustment_reason": "model_disagreement_risk",
                        "type": "model_adjusted"
                    }

                    log_signal(adjusted_signal)
                    results.append({"asset": key, "status": "adjusted", "probability": disagreement_prob})
                else:
                    results.append({"asset": key, "status": "ok", "probability": disagreement_prob})

            except Exception as e:
                results.append({"asset": key, "status": "error", "error": str(e)})

    return {"summary": results}
