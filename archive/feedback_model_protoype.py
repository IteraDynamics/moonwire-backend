# src/feedback_model_prototype.py

import json
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === 1. Pull training data from backend ===
API_URL = "https://moonwire-signal-engine-1.onrender.com/internal/generate-training-pairs"

def fetch_training_data():
    resp = requests.get(API_URL)
    if resp.status_code != 200:
        raise Exception("Failed to fetch training pairs")
    return resp.json()

# === 2. Prepare features ===
def prepare_dataset(pairs):
    rows = []
    for item in pairs:
        X = item["X"]
        rows.append({
            "score": X["score"],
            "confidence": X["confidence"],
            "label": X["label"],
            "y": item["y"],
            "weight": item["weight"]
        })

    df = pd.DataFrame(rows)

    # Encode label (y) for classification
    df["y_encoded"] = LabelEncoder().fit_transform(df["y"])

    # Encode label text as feature
    df["label_encoded"] = LabelEncoder().fit_transform(df["label"])

    return df

# === 3. Train and evaluate ===
def train_model(df):
    features = df[["score", "confidence", "label_encoded"]]
    target = df["y_encoded"]
    weights = df["weight"]

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        features, target, weights, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train, sample_weight=w_train)

    y_pred = clf.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    return clf

if __name__ == "__main__":
    print("🔍 Fetching training pairs...")
    data = fetch_training_data()
    df = prepare_dataset(data)

    if df.empty:
        print("⚠️ No data available to train.")
    else:
        print(f"✅ Loaded {len(df)} rows")
        train_model(df)
