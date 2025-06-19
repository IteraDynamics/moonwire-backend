# src/feedback_prediction_router.py

from fastapi import APIRouter
from pydantic import BaseModel
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

router = APIRouter(prefix="/internal", tags=["internal-tools"])

TRAINING_DATA_URL = "https://moonwire-signal-engine-1.onrender.com/internal/generate-training-pairs"

class SignalSnapshot(BaseModel):
    score: float
    confidence: float
    label: str

# === Mock fallback data ===
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

def load_and_train_model():
    try:
        resp = requests.get(TRAINING_DATA_URL)
        data = resp.json() if resp.status_code == 200 else []
    except Exception:
        data = []

    if len(data) < 3:
        data = mock_training_pairs

    rows = []
    for item in data:
        X = item["X"]
        rows.append({
            "score": X["score"],
            "confidence": X["confidence"],
            "label": X["label"],
            "y": item["y"],
            "weight": item["weight"]
        })

    df = pd.DataFrame(rows)
    df["y_encoded"] = LabelEncoder().fit_transform(df["y"])
    df["label_encoded"] = LabelEncoder().fit_transform(df["label"])

    X_data = df[["score", "confidence", "label_encoded"]]
    y_data = df["y_encoded"]
    weights = df["weight"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_data, y_data, sample_weight=weights)

    label_encoder = LabelEncoder().fit(df["label"])
    return model, label_encoder

@router.post("/predict-feedback-risk")
def predict_disagreement(snapshot: SignalSnapshot):
    print("[Debug] Using SignalSnapshot model")  # 🔍 Key debug line
    model, label_encoder = load_and_train_model()

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