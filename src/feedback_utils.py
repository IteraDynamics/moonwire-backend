import json
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

FEEDBACK_FILE = "data/feedback.jsonl"
MODEL_PATH = Path("models/feedback_disagreement_model.pkl")

# Confidence level mapping for fallback compatibility
CONFIDENCE_MAP = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9
}

# === Feedback Summary for Trust Score ===
def get_feedback_summary_for_signal(signal_id: str):
    if not os.path.exists(FEEDBACK_FILE):
        return {
            "num_feedback": 0,
            "num_agree": 0,
            "num_disagree": 0,
            "historical_agreement_rate": None
        }

    with open(FEEDBACK_FILE, "r") as f:
        lines = f.readlines()

    signal_feedback = [json.loads(line) for line in lines if json.loads(line).get("signal_id") == signal_id]
    if not signal_feedback:
        return {
            "num_feedback": 0,
            "num_agree": 0,
            "num_disagree": 0,
            "historical_agreement_rate": None
        }

    num_agree = sum(1 for f in signal_feedback if f.get("agree") is True)
    num_disagree = sum(1 for f in signal_feedback if f.get("agree") is False)
    num_feedback = num_agree + num_disagree

    historical_agreement_rate = num_agree / num_feedback if num_feedback > 0 else None

    return {
        "num_feedback": num_feedback,
        "num_agree": num_agree,
        "num_disagree": num_disagree,
        "historical_agreement_rate": historical_agreement_rate
    }

# === Trust Score Prediction Model (Structured) ===
def run_disagreement_prediction(score: float, confidence: float, label: str) -> float:
    model, label_encoder = load_model()

    if label not in label_encoder.classes_:
        raise ValueError(f"Unrecognized label: {label}")

    encoded_label = label_encoder.transform([label])[0]

    features = pd.DataFrame([{
        "score": score,
        "confidence": confidence,
        "label_encoded": encoded_label
    }])

    proba = model.predict_proba(features)[0]
    return max(proba)

# === Wrapper for compatibility ===
def get_disagreement_probability(label: str, score: float = 0.5, confidence="medium") -> float:
    """
    Wrapper to call disagreement prediction with minimal params.
    Handles string confidence levels by converting to float.
    """
    if isinstance(confidence, str):
        confidence = CONFIDENCE_MAP.get(confidence.lower(), 0.5)
    return run_disagreement_prediction(score=score, confidence=confidence, label=label)

# === Fallback Model Logic ===
def train_fallback_model():
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
        },
        {
            "X": {"score": 0.65, "confidence": 0.9, "label": "Bullish Momentum"},
            "y": "Accurate",
            "weight": 0.8
        }
    ]

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

def load_model():
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        # Safe hardcoded fallback for known labels including 'Bullish Momentum'
        label_encoder = LabelEncoder().fit(["Positive", "Negative", "Neutral", "Bullish Momentum"])
        return model, label_encoder
    else:
        print("[WARN] No trained model found. Using fallback mock model.")
        return train_fallback_model()