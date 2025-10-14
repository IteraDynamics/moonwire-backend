# src/model_signal_adjust_router.py

from fastapi import APIRouter
from datetime import datetime
from src.signal_log import log_signal
from src.cache_instance import cache
from src.feedback_prediction_router import predict_disagreement
from pydantic import BaseModel

router = APIRouter()

# Adjustment rules
ADJUSTMENT_FACTOR = 0.9
RISK_THRESHOLD = 0.7

# Snapshot model to match disagreement model
class SignalSnapshot(BaseModel):
    score: float
    confidence: float
    label: str

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
                continue  # Skip already-adjusted

            # Construct model input
            try:
                snapshot = SignalSnapshot(
                    score=float(latest.get("score", 0.5)),
                    confidence=float(latest.get("confidence", 0.5)),
                    label=str(latest.get("label", "Neutral"))
                )
            except Exception as e:
                results.append({"asset": key, "status": "invalid_snapshot", "error": str(e)})
                continue

            try:
                model_result = predict_disagreement(snapshot)
                disagreement_prob = model_result.get("probability", 0)

                if disagreement_prob > RISK_THRESHOLD:
                    adjusted_score = round(snapshot.score * ADJUSTMENT_FACTOR, 4)
                    adjusted_confidence = round(snapshot.confidence * ADJUSTMENT_FACTOR, 4)

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
                    results.append({
                        "asset": key,
                        "status": "adjusted",
                        "probability": disagreement_prob
                    })
                else:
                    results.append({
                        "asset": key,
                        "status": "ok",
                        "probability": disagreement_prob
                    })

            except Exception as e:
                results.append({
                    "asset": key,
                    "status": "model_error",
                    "error": str(e)
                })

    return {"summary": results}