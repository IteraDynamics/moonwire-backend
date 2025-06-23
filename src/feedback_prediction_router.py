# src/feedback_prediction_router.py

from fastapi import APIRouter
from pathlib import Path
import json
import joblib
import numpy as np

router = APIRouter()

FEEDBACK_PATH = Path("data/feedback.jsonl")
SIGNAL_LOG_PATH = Path("logs/signal_history.jsonl")

# Load your model and encoders
MODEL_PATH = Path("models/feedback_disagreement_model.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

def load_recent_signals(limit=10):
    with open(SIGNAL_LOG_PATH, "r") as f:
        lines = list(f)[-limit:]
        return [json.loads(line) for line in lines]

def predict_disagreement(snapshot):
    try:
        # Convert snapshot fields to model input
        encoded_label = label_encoder.transform([snapshot["label"]])[0]
        features = np.array([
            snapshot["score"],
            snapshot["confidence"],
            encoded_label
        ]).reshape(1, -1)
        prob = model.predict_proba(features)[0][1]  # Probability of disagreement
        return prob
    except Exception as e:
        return f"Error: {e}"

@router.get("/internal/predict-disagreements")
def predict_disagreements():
    signals = load_recent_signals()
    output = []
    for s in signals:
        if s["type"] == "raw":
            prediction = predict_disagreement(s)
            output.append({
                "signal_id": s.get("signal_id", "unknown"),
                "disagreement_probability": prediction
            })
    return {"predictions": output}