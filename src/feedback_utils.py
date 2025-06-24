import json
import os
import joblib

FEEDBACK_FILE = "data/feedback.jsonl"

def get_feedback_summary_for_signal(signal_id: str):
    if not os.path.exists(FEEDBACK_FILE):
        return {
            "num_feedback": 0,
            "num_agree": 0,
            "num_disagree": 0,
            "historical_agreement_rate": None
        }

    with open(FEEDBACK_FILE, "r") as f:
        lines = f.readlines()

    signal_feedback = [json.loads(line) for line in lines if json.loads(line).get("signal_id") == signal_id]
    if not signal_feedback:
        return {
            "num_feedback": 0,
            "num_agree": 0,
            "num_disagree": 0,
            "historical_agreement_rate": None
        }

    num_agree = sum(1 for f in signal_feedback if f.get("agree") is True)
    num_disagree = sum(1 for f in signal_feedback if f.get("agree") is False)
    num_feedback = num_agree + num_disagree

    historical_agreement_rate = num_agree / num_feedback if num_feedback > 0 else None

    return {
        "num_feedback": num_feedback,
        "num_agree": num_agree,
        "num_disagree": num_disagree,
        "historical_agreement_rate": historical_agreement_rate
    }

# Disagreement prediction model loader
_model = None
def fetch_disagreement_prediction(signal_text: str) -> float:
    global _model
    if _model is None:
        _model = joblib.load("models/disagreement_model.pkl")
    return _model.predict_proba([signal_text])[0][1]