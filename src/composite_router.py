from fastapi import APIRouter, Query
from src.signal_utils import generate_composite_signal, compute_trust_scores
from src.feedback_utils import get_feedback_summary_for_signal, run_disagreement_prediction
import os
import json

router = APIRouter(prefix="/signals")

SUPPRESSION_LOG_PATH = "data/suppression_log.jsonl"
SUPPRESSION_THRESHOLD = 0.4  # trust_score below this = suppressed

def log_suppressed_signal(signal, reason):
    log_entry = {
        "id": signal["id"],
        "asset": signal.get("asset"),
        "timestamp": signal.get("timestamp"),
        "trust_score": signal.get("trust_score"),
        "trust_label": signal.get("trust_label"),
        "reason": reason
    }
    os.makedirs(os.path.dirname(SUPPRESSION_LOG_PATH), exist_ok=True)
    with open(SUPPRESSION_LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

@router.get("/composite")
def get_composite_signal(
    asset: str = Query(...),
    twitter_score: float = Query(...),
    news_score: float = Query(...)
):
    """
    Generate a composite signal based on sentiment scores and trust insights.
    Suppresses low-trust signals and flags likely disagreement.
    """
    signal = generate_composite_signal(asset, twitter_score, news_score)

    feedback_summary = get_feedback_summary_for_signal(signal["id"])

    # Fix: handle string-to-float safely using a mapping
    confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
    confidence_raw = signal.get("confidence", "").lower()
    confidence_value = confidence_map.get(confidence_raw, 0.5)

    predicted_disagreement_prob = run_disagreement_prediction(
        score=signal["score"],
        confidence=confidence_value,
        label=signal["label"]
    )

    trust_insights = {
        signal["id"]: {
            "historical_agreement_rate": feedback_summary.get("historical_agreement_rate"),
            "predicted_disagreement_prob": predicted_disagreement_prob
        }
    }

    compute_trust_scores(signal, trust_insights)

    if signal.get("trust_score", 0.5) < SUPPRESSION_THRESHOLD:
        log_suppressed_signal(signal, reason="trust_score_below_threshold")

    return signal