from datetime import datetime
import uuid
from collections import defaultdict
import json
import os

# === Utility Functions ===
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

# === Trust Score Logic ===
def confidence_to_float(conf):
    mapping = {"low": 0.3, "medium": 0.6, "high": 0.9}
    if isinstance(conf, (int, float)):
        return float(conf)
    return mapping.get(str(conf).lower(), 0.5)

def compute_trust_scores(disagreement_predict_fn):
    signal_log_path = "logs/signal_history.jsonl"
    feedback_path = "data/feedback.jsonl"

    if not os.path.exists(signal_log_path):
        return []

    signal_map = {}
    with open(signal_log_path, "r") as f:
        for line in f:
            try:
                s = json.loads(line)
                signal_map[s["signal_id"]] = s
            except json.JSONDecodeError:
                continue

    # Load feedback counts
    feedback_counts = defaultdict(lambda: {"agree": 0, "disagree": 0})
    if os.path.exists(feedback_path):
        with open(feedback_path, "r") as f:
            for line in f:
                try:
                    fb = json.loads(line)
                    sid = fb.get("signal_id")
                    if sid:
                        if fb.get("agree") is True:
                            feedback_counts[sid]["agree"] += 1
                        else:
                            feedback_counts[sid]["disagree"] += 1
                except json.JSONDecodeError:
                    continue

    results = []
    for sid, signal in signal_map.items():
        score = signal.get("score", 0)
        confidence = signal.get("confidence", 0.5)
        label = signal.get("label", "Neutral")

        disagreement = disagreement_predict_fn({
            "score": score,
            "confidence": confidence,
            "label": label
        })
        disagreement_prob = disagreement.get("probability", 0.5)

        counts = feedback_counts.get(sid, {"agree": 0, "disagree": 0})
        total = counts["agree"] + counts["disagree"]
        agreement_rate = counts["agree"] / total if total > 0 else 0.5

        conf = confidence_to_float(confidence)
        trust_score = conf * (1 - disagreement_prob) * agreement_rate

        if trust_score > 0.7:
            trust_label = "High Trust"
        elif trust_score > 0.4:
            trust_label = "Moderate Trust"
        else:
            trust_label = "Low Trust"

        results.append({
            "signal_id": sid,
            "trust_score": round(trust_score, 3),
            "trust_label": trust_label,
            "predicted_disagreement_prob": disagreement_prob,
            "agreement_rate": round(agreement_rate, 3),
            "feedback_summary": counts
        })

    return results