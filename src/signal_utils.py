import json
from collections import defaultdict
from pathlib import Path

FEEDBACK_PATH = Path("data/feedback.jsonl")
SIGNAL_HISTORY_PATH = Path("logs/signal_history.jsonl")


def load_feedback_summary():
    summary = defaultdict(lambda: {"agree": 0, "disagree": 0})
    if FEEDBACK_PATH.exists():
        with FEEDBACK_PATH.open() as f:
            for line in f:
                entry = json.loads(line)
                signal_id = entry.get("signal_id")
                feedback = entry.get("user_feedback")
                if signal_id and feedback:
                    summary[signal_id][feedback] += 1
    return summary


def calculate_agreement_rate(feedback_summary):
    rates = {}
    for signal_id, counts in feedback_summary.items():
        agree = counts["agree"]
        disagree = counts["disagree"]
        total = agree + disagree
        if total > 0:
            rates[signal_id] = agree / total
        else:
            rates[signal_id] = 0.5  # Default neutral value
    return rates


def normalize_score(score):
    return max(0.0, min(1.0, round(score, 3)))


def compute_trust_scores(disagreement_predictor):
    feedback_summary = load_feedback_summary()
    agreement_rates = calculate_agreement_rate(feedback_summary)

    trust_insights = []

    if SIGNAL_HISTORY_PATH.exists():
        with SIGNAL_HISTORY_PATH.open() as f:
            for line in f:
                entry = json.loads(line)
                signal_id = entry.get("signal_id")
                confidence = entry.get("confidence", 0.5)
                score = entry.get("score", 0.5)
                label = entry.get("label", "Neutral")

                agreement_weight = agreement_rates.get(signal_id, 0.5)

                payload = {
                    "score": score,
                    "confidence": confidence,
                    "label": label
                }
                prediction = disagreement_predictor(payload)
                disagreement_prob = prediction.get("probability", 0.5)

                trust_score = normalize_score(
                    confidence * (1 - disagreement_prob) * agreement_weight
                )

                if trust_score < 0.4:
                    trust_label = "Low Trust"
                elif trust_score < 0.75:
                    trust_label = "Moderate"
                else:
                    trust_label = "High"

                trust_insights.append({
                    "signal_id": signal_id,
                    "trust_score": trust_score,
                    "trust_label": trust_label,
                    "predicted_disagreement_prob": disagreement_prob,
                    "agreement_rate": round(agreement_weight, 3),
                    "feedback_summary": feedback_summary.get(signal_id, {"agree": 0, "disagree": 0})
                })

    return trust_insights
