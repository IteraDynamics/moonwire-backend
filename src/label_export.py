# src/label_export.py

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# === Mock signal_history.jsonl fallback ===
mock_feedback = [
    {
        "type": "user_feedback",
        "timestamp": "2025-06-08T18:00:00Z",
        "asset": "BTC",
        "sentiment": 0.45,
        "user_feedback": "Too bullish",
        "confidence": 0.9,
        "context": "Disagrees with price trend"
    },
    {
        "type": "user_feedback",
        "timestamp": "2025-06-08T19:00:00Z",
        "asset": "ETH",
        "sentiment": 0.35,
        "user_feedback": "Neutral is fair",
        "confidence": 0.6,
        "context": "Aligned with recent news"
    },
    {
        "type": "user_feedback",
        "timestamp": "2025-06-08T20:00:00Z",
        "asset": "BTC",
        "sentiment": 0.42,
        "user_feedback": "Too optimistic",
        "confidence": 0.3,
        "context": "Volume dropping, sentiment rising"
    }
]

# === Reliability scoring logic ===
def compute_reliability(confidence: float) -> float:
    return round(confidence * 0.9 + 0.1, 3)

# === Export logic ===
def export_labels_to_csv(feedback_data, out_path: str = "logs/feedback_labels.csv"):
    rows = []

    for entry in feedback_data:
        if entry.get("type") != "user_feedback":
            continue

        rows.append({
            "asset": entry["asset"],
            "timestamp": entry["timestamp"],
            "user_feedback": entry["user_feedback"],
            "confidence": entry["confidence"],
            "reliability_score": compute_reliability(entry["confidence"]),
            "context": entry.get("context", "")
        })

    df = pd.DataFrame(rows)
    Path("logs").mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[✔] Exported {len(df)} label rows to {out_path}")

if __name__ == "__main__":
    export_labels_to_csv(mock_feedback)
