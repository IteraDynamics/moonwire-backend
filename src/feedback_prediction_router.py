# src/feedback_prediction_router.py

from fastapi import APIRouter, Request
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

def load_and_train_model():
    resp = requests.get(TRAINING_DATA_URL)
    if resp.status_code != 200:
        return None, "Failed to fetch training data"

    data = resp.json()
    if len(data) < 3:
        return None, "Insufficient training samples"

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
    return (model, label_encoder)

@router.post("/predict-feedback-risk")
def predict_disagreement(snapshot: SignalSnapshot):
    model, label_encoder_or_error = load_and_train_model()

    if model is None:
        return {"status": "error", "detail": label_encoder_or_error}

    encoded_label = label_encoder_or_error.transform([snapshot.label])[0]

    features = pd.DataFrame([{
        "score": snapshot.score,
        "confidence": snapshot.confidence,
        "label_encoded": encoded_label
    }])

    pred_proba = model.predict_proba(features)
    predicted_class = model.predict(features)[0]
    disagree_prob = round(max(pred_proba[0]), 3)

    return {
        "likely_disagreed": bool(predicted_class),
        "probability": disagree_prob
    }
