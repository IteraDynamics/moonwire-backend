import json
from collections import defaultdict
from pathlib import Path

FEEDBACK_PATH = Path("data/feedback.jsonl")

def load_feedback_data():
    feedback_entries = []
    if FEEDBACK_PATH.exists():
        with open(FEEDBACK_PATH, "r") as f:
            for line in f:
                try:
                    feedback_entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue  # skip malformed lines
    return feedback_entries

def get_feedback_summary_for_signal(signal_id):
    feedback_data = load_feedback_data()
    summary = defaultdict(int)

    for entry in feedback_data:
        if entry.get("signal_id") == signal_id:
            vote = entry.get("vote")
            if vote == "agree":
                summary["agree"] += 1
            elif vote == "disagree":
                summary["disagree"] += 1

    return dict(summary)

def compute_agreement_rate(summary):
    agree = summary.get("agree", 0)
    disagree = summary.get("disagree", 0)
    total = agree + disagree
    return round(agree / total, 3) if total > 0 else None