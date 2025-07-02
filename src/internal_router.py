from fastapi import APIRouter
from pydantic import BaseModel
from collections import defaultdict, Counter
from typing import List
import json
import os
import requests
from src.signal_utils import compute_trust_scores

router = APIRouter(prefix="/internal", tags=["internal-tools"])

FEEDBACK_LOG_PATH = "data/feedback.jsonl"  # Location of feedback data
SUPPRESSION_LOG_PATH = "data/suppression_log.jsonl"

# === Feedback Summary Route ===
@router.get("/feedback-summary")
def get_feedback_summary():
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return {
            "total_feedback": 0,
            "agree_percentage": 0.0,
            "disagree_count": 0,
            "most_disagreed_signals": []
        }

    total = 0
    agree = 0
    disagree_signals = defaultdict(list)

    with open(FEEDBACK_LOG_PATH, "r") as f:
        for line in f:
            try:
                fb = json.loads(line)
                total += 1
                if fb.get("agree") is True:
                    agree += 1
                else:
                    sid = fb.get("signal_id", "unknown")
                    disagree_signals[sid].append(fb.get("note", ""))
            except json.JSONDecodeError:
                continue

    # Get top 3 most disagreed signals
    top_signals = sorted(disagree_signals.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    most_disagreed = [
        {
            "signal_id": sid,
            "count": len(notes),
            "notes": [n for n in notes if n]
        }
        for sid, notes in top_signals
    ]

    return {
        "total_feedback": total,
        "agree_percentage": round((agree / total) * 100, 2) if total > 0 else 0.0,
        "disagree_count": total - agree,
        "most_disagreed_signals": most_disagreed
    }


# === Signal Trust Score Route ===
@router.get("/signal-trust-insights")
def signal_trust_insights():
    def fetch_disagreement_prediction(payload):
        response = requests.post(
            "http://localhost:8000/internal/predict-feedback-risk",
            json=payload
        )
        if response.status_code == 200:
            return response.json()
        return {"probability": 0.5}

    return compute_trust_scores(fetch_disagreement_prediction)


# === Suppression Summary Route ===
@router.get("/suppression-summary")
def suppression_summary():
    if not os.path.exists(SUPPRESSION_LOG_PATH):
        return {"total_suppressed": 0, "by_asset": {}, "by_trust_band": {}}

    asset_counts = defaultdict(int)
    trust_band_counts = defaultdict(int)
    total = 0

    with open(SUPPRESSION_LOG_PATH, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                total += 1
                asset = entry.get("asset", "Unknown")
                trust_score = entry.get("trust_score", 0.5)

                asset_counts[asset] += 1

                if trust_score < 0.2:
                    band = "< 0.2"
                elif trust_score < 0.4:
                    band = "0.2–0.4"
                else:
                    band = "0.4+"

                trust_band_counts[band] += 1

            except json.JSONDecodeError:
                continue

    return {
        "total_suppressed": total,
        "by_asset": dict(asset_counts),
        "by_trust_band": dict(trust_band_counts)
    }