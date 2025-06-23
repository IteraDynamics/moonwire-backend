from fastapi import APIRouter
from pathlib import Path
import json

router = APIRouter()
LOG_FILE = Path("logs/signal_history.jsonl")

@router.get("/internal/high-disagreement-summary")
def get_high_disagreement_signals():
    """
    Scans the signal history for model_adjusted entries caused by disagreement risk
    and returns a summary grouped by asset.
    """
    if not LOG_FILE.exists():
        return {"summary": []}

    result = {}
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                signal = json.loads(line)
                if (
                    signal.get("type") == "model_adjusted" and
                    signal.get("adjustment_reason") == "model_disagreement_risk"
                ):
                    asset = signal.get("asset", "unknown")
                    if asset not in result:
                        result[asset] = []
                    result[asset].append({
                        "timestamp": signal.get("timestamp"),
                        "score": signal.get("score"),
                        "confidence": signal.get("confidence"),
                        "adjusted_at": signal.get("adjusted_at")
                    })
            except json.JSONDecodeError:
                continue

    summary = [{"asset": k, "adjusted_signals": v} for k, v in result.items()]
    return {"summary": summary}