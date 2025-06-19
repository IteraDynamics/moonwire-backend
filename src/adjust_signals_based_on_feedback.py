# src/adjust_signals_based_on_feedback.py

import json
from pathlib import Path
from datetime import datetime
import requests
from src.signal_log import log_signal

LOG_FILE = Path("logs/signal_history.jsonl")
PREDICT_API = "https://moonwire-signal-engine-1.onrender.com/internal/predict-feedback-risk" # change for deployment

def load_signals():
    if not LOG_FILE.exists():
        print("[Debug] Log file not found.")
        return []
    with open(LOG_FILE, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def get_disagreement_prediction(signal):
    payload = {
        "score": signal.get("score", 0),
        "confidence": signal.get("confidence", 0.5),
        "label": signal.get("label", "Neutral"),
        "fallback_type": signal.get("fallback_type", "unknown")
    }
    try:
        response = requests.post(PREDICT_API, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[Error] Prediction failed: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"[Error] Prediction error: {e}")
        return None

def adjust_signals():
    all_signals = load_signals()
    if not all_signals:
        print("[Debug] No signals loaded.")
        return {"summary": []}

    latest_by_asset = {}
    for s in all_signals:
        if s.get("type", "raw") == "raw":
            latest_by_asset[s["asset"]] = s

    summary = []

    for asset, signal in latest_by_asset.items():
        prediction = get_disagreement_prediction(signal)
        if not prediction:
            continue

        prob = prediction.get("probability", 0)
        if prob > 0.7:
            adjusted_conf = max(signal.get("confidence", 0.5) - 0.1, 0)
            new_signal = {
                **signal,
                "confidence": adjusted_conf,
                "type": "model_adjusted",
                "adjustment_reason": "model_disagreement_risk",
                "adjusted_at": datetime.utcnow().isoformat()
            }
            log_signal(signal_data=new_signal)
            status = "adjusted"
        else:
            status = "ok"

        summary.append({
            "asset": asset,
            "status": status,
            "probability": round(prob, 3)
        })

    return {"summary": summary}