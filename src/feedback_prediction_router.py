# src/feedback_prediction_router.py

from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

router = APIRouter(prefix="/internal", tags=["internal-tools"])

MODEL_PATH = Path("models/feedback_disagreement_model.pkl")

class SignalSnapshot(BaseModel):
    score: float
    confidence: float
    label: str

# === Fallback mock training data ===
mock_training_pairs = [
    {
        "X": {"score": 0.4, "confidence": 0.8, "label": "Positive"},
        "y": "Too bearish",
        "weight": 0.7
    },
    {
        "X": {"score": 0.7, "confidence": 0.9, "label": "Positive"},
        "y": "Accurate",
        "weight": 0.9
    },
    {
        "X": {"score": 0.2, "confidence": 0.6, "label": "Negative"},
        "y": "Too bullish",
        "weight": 0.6
    },
    {
        "X": {"score": 0.5, "confidence": 0.7, "label": "Neutral"},
        "y": "Too bearish",
        "weight": 0.75
    }
]

# === Fallback training function if model not found ===
def train_fallback_model():
    rows = []
    for item in mock_training_pairs:
        X = item["X"]
        rows.append({
            "score": X["score"],
            "confidence": X["confidence"],
            "label": X["label"],
            "y": item["y"],
            "weight": item["weight"]
        })

    df = pd.DataFrame(rows)
    label_encoder = LabelEncoder().fit(df["label"])
    df["label_encoded"] = label_encoder.transform(df["label"])
    df["y_encoded"] = LabelEncoder().fit_transform(df["y"])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[["score", "confidence", "label_encoded"]], df["y_encoded"], sample_weight=df["weight"])

    return model, label_encoder

# === Model loader ===
def load_model():
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        label_encoder = LabelEncoder().fit(["Positive", "Negative", "Neutral"])  # Must match training
        return model, label_encoder
    else:
        print("[WARN] No trained model found. Using fallback mock model.")
        return train_fallback_model()

@router.post("/predict-feedback-risk")
def predict_disagreement(snapshot: SignalSnapshot):
    model, label_encoder = load_model()
    encoded_label = label_encoder.transform([snapshot.label])[0]
    features = pd.DataFrame([{
        "score": snapshot.score,
        "confidence": snapshot.confidence,
        "label_encoded": encoded_label
    }])

    proba = model.predict_proba(features)[0]
    predicted_class = model.predict(features)[0]

    return {
        "likely_disagreed": bool(predicted_class),
        "probability": round(max(proba), 3)
    }