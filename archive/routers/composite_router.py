from fastapi import APIRouter, Query
from src.signal_utils import generate_composite_signal, compute_trust_scores
from src.feedback_utils import get_feedback_summary_for_signal, run_disagreement_prediction
from datetime import datetime, timedelta
from pathlib import Path
import os
import json

router = APIRouter(prefix="/signals")

SUPPRESSION_LOG_PATH = "data/suppression_review_queue.jsonl"
SUPPRESSION_THRESHOLD = 0.4  # trust_score below this = suppressed
SIGNAL_HISTORY_PATH = Path("data/signal_history.jsonl")

def get_recent_suppressions(asset: str, lookback_minutes=1440):
    """
    Returns number of times this asset was suppressed in past lookback_minutes.
    """
    if not os.path.exists(SUPPRESSION_LOG_PATH):
        return 0

    count = 0
    cutoff = datetime.utcnow() - timedelta(minutes=lookback_minutes)

    with open(SUPPRESSION_LOG_PATH, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if (
                    entry.get("asset") == asset and
                    datetime.fromisoformat(entry.get("timestamp")) >= cutoff
                ):
                    count += 1
            except Exception:
                continue
    return count

def add_retrain_hint_if_applicable(signal: dict, log_entry: dict):
    """
    Adds retrain_hint to log_entry if signal matches known retraining patterns.
    """
    hints = []

    if signal.get("confidence") == "low":
        hints.append("low_confidence")

    if signal.get("fallback_type") == "missing_agreement":
        hints.append("missing_agreement")

    if get_recent_suppressions(signal.get("asset")) >= 2:
        hints.append("asset_spike")

    if hints:
        log_entry["retrain_hint"] = hints[0]  # Prioritize first match

def log_suppressed_signal(signal, reason):
    log_entry = {
        "id": signal["id"],
        "asset": signal.get("asset"),
        "timestamp": signal.get("timestamp"),
        "trust_score": signal.get("trust_score"),
        "trust_label": signal.get("trust_label"),
        "reason": reason,
        "status": "pending",
        "full_payload": signal
    }

    add_retrain_hint_if_applicable(signal, log_entry)

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
    Suppresses low-trust signals and logs them for internal review.
    Also appends the full signal to signal_history.jsonl for analytics use.
    """
    signal = generate_composite_signal(asset, twitter_score, news_score)

    feedback_summary = get_feedback_summary_for_signal(signal["id"])
    confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}

    try:
        predicted_disagreement_prob = run_disagreement_prediction(
            score=signal["score"],
            confidence=confidence_map[signal["confidence"]],
            label=signal["label"]
        )
    except Exception:
        predicted_disagreement_prob = 0.9  # Assume high disagreement if prediction fails
        signal["fallback_type"] = "disagreement_prediction_failed"

    trust_insights = {
        signal["id"]: {
            "historical_agreement_rate": feedback_summary.get("historical_agreement_rate"),
            "predicted_disagreement_prob": predicted_disagreement_prob
        }
    }

    compute_trust_scores(signal, trust_insights)

    # Inject fallback trust score if not present
    if "trust_score" not in signal or signal["trust_score"] is None:
        signal["trust_score"] = 0.2  # Force low score to test suppression
        signal["trust_label"] = "Low"
        signal["fallback_type"] = "forced_low_for_testing"

    if signal.get("trust_score", 0.5) < SUPPRESSION_THRESHOLD:
        log_suppressed_signal(signal, reason="trust_score_below_threshold")

    # ✅ Append to signal_history.jsonl
    try:
        SIGNAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SIGNAL_HISTORY_PATH.open("a") as f:
            f.write(json.dumps(signal) + "\n")
    except Exception as e:
        print(f"[⚠️] Failed to append signal to history: {e}")

    return signal