from datetime import datetime
import uuid


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
        return "Bullish Momentum"
    elif score <= -0.6:
        return "Bearish Reversal"
    else:
        return "Neutral Drift"


def get_trend(score):
    if score > 0:
        return "upward"
    elif score < 0:
        return "downward"
    else:
        return "flat"


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


# === Trust Score Computation Utilities ===
import json
import os
from src.feedback_utils import get_feedback_summary_for_signal

def compute_trust_scores(prediction_fn):
    def inner(signals):
        results = []
        for sig in signals:
            score = sig.get("score")
            signal_id = sig.get("id") or sig.get("signal_id")

            disagreement_prob = prediction_fn(signal_id)
            confidence = sig.get("confidence", "low")
            confidence_val = 1.0 if confidence == "high" else 0.6 if confidence == "medium" else 0.3

            feedback = get_feedback_summary_for_signal(signal_id)
            agree = feedback.get("agree", 0)
            disagree = feedback.get("disagree", 0)
            total = agree + disagree
            agreement_rate = agree / total if total else 0.5

            agreement_weight = 0.5
            confidence_weight = 0.3
            disagreement_weight = 0.2

            trust_score = (
                confidence_val * confidence_weight +
                (1 - disagreement_prob) * disagreement_weight +
                agreement_rate * agreement_weight
            )

            # 🛠️ Trust label classification logic
            if trust_score >= 0.75:
                trust_label = "High Trust"
            elif trust_score >= 0.4:
                trust_label = "Medium Trust"
            else:
                trust_label = "Low Trust"

            results.append({
                "signal_id": signal_id,
                "trust_score": round(trust_score, 3),
                "trust_label": trust_label,
                "predicted_disagreement_prob": round(disagreement_prob, 2),
                "agreement_rate": round(agreement_rate, 3),
                "feedback_summary": feedback
            })

        return results
    return inner