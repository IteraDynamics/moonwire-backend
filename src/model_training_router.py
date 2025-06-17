# src/model_training_router.py

from fastapi import APIRouter
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from collections import Counter

router = APIRouter(prefix="/internal", tags=["internal-tools"])

API_URL = "https://moonwire-signal-engine-1.onrender.com/internal/generate-training-pairs"

@router.post("/train-feedback-model")
def train_feedback_model():
    # === Fetch training data from internal route ===
    resp = requests.get(API_URL)
    if resp.status_code != 200:
        return {"error": "Failed to fetch training data"}

    data = resp.json()
    if not data:
        return {"status": "no training data"}

    # === Prepare data frame ===
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
    if len(df) < 2:
        return {"status": "not enough data to train", "samples": len(df)}

    # === Encode features and labels ===
    df["y_encoded"] = LabelEncoder().fit_transform(df["y"])
    df["label_encoded"] = LabelEncoder().fit_transform(df["label"])

    X_data = df[["score", "confidence", "label_encoded"]]
    y_data = df["y_encoded"]
    weights = df["weight"]

    # === Train/test split ===
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_data, y_data, weights, test_size=0.3, random_state=42
    )

    # === Train classifier ===
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train, sample_weight=w_train)

    y_pred = clf.predict(X_test)
    f1 = round(f1_score(y_test, y_pred, average="macro"), 3)
    class_counts = Counter(df["y"])

    return {
        "status": "trained",
        "samples_used": len(df),
        "f1_macro": f1,
        "class_distribution": dict(class_counts)
    }
