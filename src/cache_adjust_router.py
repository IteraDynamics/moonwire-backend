# src/cache_adjust_router.py

from fastapi import APIRouter
from datetime import datetime
from src.cache_instance import cache
from src.feedback_prediction_router import predict_disagreement
from pydantic import BaseModel

router = APIRouter(prefix="/internal", tags=["internal-tools"])

RISK_THRESHOLD = 0.7
ADJUSTMENT_FACTOR = 0.9

class SignalSnapshot(BaseModel):
    score: float
    confidence: float
    label: str

@router.post("/adjust-cached-signals")
def adjust_cached_signals():
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

        try:
            snapshot = SignalSnapshot(
                score=latest["score"],
                confidence=latest["confidence"],
                label=latest["label"]
            )
        except Exception as e:
            summary.append({"asset": key, "status": "error", "reason": str(e)})
            continue

        result = predict_disagreement(snapshot)
        prob = result.get("probability", 0)

        if prob > RISK_THRESHOLD:
            summary.append({
                "asset": key.replace("_history", ""),
                "status": "high-risk",
                "probability": prob,
                "adjustment_applied": "not-written"
            })
        else:
            summary.append({
                "asset": key.replace("_history", ""),
                "status": "ok",
                "probability": prob
            })

    return {"summary": summary}