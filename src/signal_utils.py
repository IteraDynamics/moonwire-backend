# src/signal_utils.py

from datetime import datetime, timedelta
import uuid
import json
import os
from src.feedback_utils import run_disagreement_prediction

SUPPRESSION_REVIEW_PATH = "data/suppression_review_queue.jsonl"

def same_sign(a, b):
    return (a >= 0 and b >= 0) or (a < 0 and b < 0)

def blend_scores(twitter_score, news_score, twitter_weight=0.6, news_weight=0.4):
    raw_score = (twitter_score * twitter_weight) + (news_score * news_weight)
    if same_sign(twitter_score, news_score):
        return raw_score
    else:
        return raw_score * 0.5  # dampens disagreement

def determine_confidence(score, twitter_score, news_score):
    if abs(score) >= 0.6 and same_sign(twitter_score, news_score):
        return "high"
    elif abs(score) >= 0.4:
        return "medium"
    else:
        return "low"

def label_signal(score):
    if score >= 0.6:
        return "Positive"
    elif score <= -0.6:
        return "Negative"
    else:
        return "Neutral"

def get_trend(score):
    if score > 0:
        return "upward"
    elif score < 0:
        return "downward"
    else:
        return "flat"

def compute_trust_scores(signal, trust_insights):
    insight = trust_insights.get(signal["id"], {})

    agreement = insight.get("historical_agreement_rate")
    disagreement_prob = insight.get("predicted_disagreement_prob")

    fallback_used = False

    if agreement is None:
        agreement = 0.5
        signal["fallback_type"] = "missing_agreement"
        fallback_used = True

    if disagreement_prob is None:
        try:
            disagreement_prob = run_disagreement_prediction(
                score=signal["score"],
                confidence={"low": 0.3, "medium": 0.6, "high": 0.9}[signal["confidence"]],
                label=signal["label"]
            )
        except Exception as e:
            disagreement_prob = 0.5
            signal["fallback_type"] = "missing_disagreement"
            fallback_used = True

    trust_score = round(
        0.6 * agreement + 0.4 * (1 - disagreement_prob),
        3
    )

    signal["trust_score"] = trust_score

    if trust_score >= 0.75:
        signal["trust_label"] = "Trusted"
    elif trust_score <= 0.35:
        signal["trust_label"] = "Untrusted"
    else:
        signal["trust_label"] = "Uncertain"

    if fallback_used:
        log_to_review_queue(signal, reason=signal.get("fallback_type", "trust_fallback"))

def detect_retrain_hint(signal):
    hints = []

    if signal.get("confidence") == "low":
        hints.append("low_confidence")

    if signal.get("fallback_type") == "missing_agreement":
        hints.append("missing_agreement")

    # Optional: check for asset spike (>=2 in last 24h)
    recent_signals = []
    try:
        if os.path.exists(SUPPRESSION_REVIEW_PATH):
            with open(SUPPRESSION_REVIEW_PATH, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    if (
                        entry["asset"] == signal["asset"]
                        and datetime.fromisoformat(entry["timestamp"]) >= datetime.utcnow() - timedelta(hours=24)
                    ):
                        recent_signals.append(entry)
        if len(recent_signals) >= 2:
            hints.append("asset_spike")
    except Exception:
        pass

    return hints[0] if hints else None

def log_to_review_queue(signal, reason):
    entry = {
        "id": signal["id"],
        "asset": signal["asset"],
        "timestamp": signal["timestamp"],
        "score": signal["score"],
        "confidence": signal["confidence"],
        "label": signal["label"],
        "trust_score": signal.get("trust_score", 0.5),
        "trust_label": signal.get("trust_label", "Uncertain"),
        "reason": reason,
        "status": "pending"
    }

    hint = detect_retrain_hint(signal)
    if hint:
        entry["retrain_hint"] = hint

    os.makedirs(os.path.dirname(SUPPRESSION_REVIEW_PATH), exist_ok=True)
    with open(SUPPRESSION_REVIEW_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def generate_composite_signal(asset, twitter_score, news_score, timestamp=None):
    score = blend_scores(twitter_score, news_score)
    return {
        "id": f"sig_{uuid.uuid4().hex[:8]}",
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "asset": asset,
        "score": round(score, 4),
        "confidence": determine_confidence(score, twitter_score, news_score),
        "label": label_signal(score),
        "trend": get_trend(score),
        "top_drivers": ["twitter sentiment", "news sentiment"],
        "fallback_type": None
    }