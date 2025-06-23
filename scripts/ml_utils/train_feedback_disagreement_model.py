

# scripts/train_feedback_disagreement_model.py

import json
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Paths
FEEDBACK_PATH = Path("data/retrain_queue.jsonl")
MODEL_OUTPUT_PATH = Path("models/feedback_disagreement_model.pkl")

def load_training_data():
    if not FEEDBACK_PATH.exists():
        print("⚠️ No retraining data found at", FEEDBACK_PATH)
        return pd.DataFrame()

    rows = []
    with open(FEEDBACK_PATH, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                rows.append(entry)
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(rows)

def preprocess_data(df):
    if df.empty:
        return None, None

    df = df.dropna(subset=["score", "confidence", "label", "agree"])
    df["label_encoded"] = LabelEncoder().fit_transform(df["label"])
    X = df[["score", "confidence", "label_encoded"]]
    y = (~df["agree"]).astype(int)  # 1 = disagreement, 0 = agreement
    return X, y

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def save_model(model):
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"✅ Model saved to {MODEL_OUTPUT_PATH}")

def main():
    print("🚀 Training feedback disagreement model...")
    df = load_training_data()
    X, y = preprocess_data(df)

    if X is None or y is None or len(X) < 5:
        print("❌ Not enough valid data to train.")
        return

    model = train_model(X, y)
    save_model(model)
    print("🏁 Done.")

if __name__ == "__main__":
    main()