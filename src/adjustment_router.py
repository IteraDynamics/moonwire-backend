from fastapi import APIRouter
from datetime import datetime
import json
from pathlib import Path
import requests
import uuid

LOG_FILE = Path("logs/signal_history.jsonl")
PREDICT_URL = "http://localhost:8000/internal/predict-feedback-risk"  # ✅ Swap with Render URL if needed

router = APIRouter()

@router.post("/internal/adjust-signals-based-on-feedback")
def adjust_signals():
    if not LOG_FILE.exists():
        return {"summary": []}

    adjustments = []
    new_entries = []

    # Load signals
    with open(LOG_FILE, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    for row in entries:
        if row.get("type") != "raw":
            continue

        model_input = {
            "score": row.get("score", 0.5),
            "confidence": row.get("confidence", 0.5),
            "label": row.get("label", "Neutral"),
            "fallback_type": row.get("fallback_type", "unknown")
        }

        try:
            r = requests.post(PREDICT_URL, json=model_input)
            r.raise_for_status()
            result = r.json()
        except Exception as e:
            adjustments.append({
                "asset": row.get("asset"),
                "status": "error",
                "reason": str(e)
            })
            continue

        probability = result.get("probability", 0)
        if probability > 0.7:
            adjusted_score = max(row.get("score", 0.5) - 0.05, 0.0)
            adjusted_conf = max(row.get("confidence", 0.5) - 0.05, 0.0)

            new_entry = {
                **row,
                "id": f"adjusted_{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.utcnow().isoformat(),
                "adjustment_applied": True,
                "adjustment_reason": "model_disagreement_risk",
                "score": round(adjusted_score, 4),
                "confidence": round(adjusted_conf, 4),
                "type": "adjusted"
            }

            new_entries.append(new_entry)

            adjustments.append({
                "asset": row.get("asset"),
                "status": "adjusted",
                "probability": round(probability, 3),
                "adjustment_reason": "model_disagreement_risk"
            })
        else:
            adjustments.append({
                "asset": row.get("asset"),
                "status": "ok",
                "probability": round(probability, 3)
            })

    if new_entries:
        with open(LOG_FILE, "a") as f:
            for entry in new_entries:
                f.write(json.dumps(entry) + "\n")

    return {"summary": adjustments}