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

def train_and_save_model():
    df = load_training_data()

    if df.empty or 'note' not in df.columns or 'asset' not in df.columns:
        print("⚠️ Not enough data to train.")
        return

    df["label"] = df["agree"].apply(lambda x: "agree" if x else "disagree")
    df = df[["asset", "note", "label"]].dropna()

    X = df[["asset", "note"]]
    y = df["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    vectorizer = lambda row: f"{row['asset']} {row['note']}".lower()
    X_vectorized = X.apply(vectorizer, axis=1)

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X_vectorized)

    model = RandomForestClassifier()
    model.fit(X_tfidf, y_encoded)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "tfidf": tfidf,
        "label_encoder": label_encoder
    }, MODEL_OUTPUT_PATH)

    print("✅ Model trained and saved to", MODEL_OUTPUT_PATH)

def predict_disagreement(payload):
    if not MODEL_OUTPUT_PATH.exists():
        raise FileNotFoundError("Trained model file not found.")

    data = joblib.load(MODEL_OUTPUT_PATH)
    model = data["model"]
    tfidf = data["tfidf"]
    label_encoder = data["label_encoder"]

    text = f"{payload['asset']} {payload.get('note', '')}".lower()
    X_input = tfidf.transform([text])

    probas = model.predict_proba(X_input)[0]
    disagree_index = list(label_encoder.classes_).index("disagree")
    disagree_prob = probas[disagree_index]

    return {
        "likely_disagreed": disagree_prob >= 0.5,
        "probability": round(disagree_prob, 2)
    }