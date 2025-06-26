# src/disagreement_utils.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from pydantic import BaseModel

MODEL_PATH = Path("models/feedback_disagreement_model.pkl")

class SignalSnapshot(BaseModel):
    score: float
    confidence: float
    label: str

CONFIDENCE_THRESHOLD = 0.5

def predict_disagreement(snapshot: SignalSnapshot):
    model, label_encoder = load_model()
    encoded_label = label_encoder.transform([snapshot.label])[0]

    features = pd.DataFrame([{
        "score": snapshot.score,
        "confidence": snapshot.confidence,
        "label_encoded": encoded_label
    }])

    proba = model.predict_proba(features)[0]
    predicted_prob = max(proba)
    likely_disagreed = predicted_prob > CONFIDENCE_THRESHOLD

    return {
        "likely_disagreed": likely_disagreed,
        "probability": predicted_prob
    }

def train_fallback_model():
    fallback_data = [
        {"score": 0.3, "confidence": 0.9, "label": "Positive", "y": "Too bearish", "weight": 0.8},
        {"score": 0.7, "confidence": 0.6, "label": "Negative", "y": "Too bullish", "weight": 0.7},
        {"score": 0.5, "confidence": 0.7, "label": "Neutral", "y": "Accurate", "weight": 0.9}
    ]

    df = pd.DataFrame(fallback_data)
    label_encoder = LabelEncoder().fit(df["label"])
    df["label_encoded"] = label_encoder.transform(df["label"])
    df["y_encoded"] = LabelEncoder().fit_transform(df["y"])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[["score", "confidence", "label_encoded"]], df["y_encoded"], sample_weight=df["weight"])

    return model, label_encoder

def load_model():
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        label_encoder = LabelEncoder().fit(["Positive", "Negative", "Neutral"])
        return model, label_encoder
    else:
        print("[WARN] No trained model found. Using fallback model.")
        return train_fallback_model()

# ✅ ADD THIS WRAPPER FOR EXTERNAL CALLERS

def get_disagreement_probability(label: str, score: float, confidence: float) -> float:
    """
    Wrapper function that builds a snapshot and extracts the predicted
    disagreement probability.
    """
    snapshot = SignalSnapshot(label=label, score=score, confidence=confidence)
    result = predict_disagreement(snapshot)
    return result.get("probability", 0.5)