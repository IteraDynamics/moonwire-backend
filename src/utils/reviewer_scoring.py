# src/utils/reviewer_scoring.py

import json
from collections import defaultdict
from typing import List, Dict

def load_logs(path: str) -> List[dict]:
    with open(path, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def compute_reviewer_scores(logs: List[dict]) -> List[Dict]:
    reviewers = defaultdict(list)

    for entry in logs:
        if "reviewer_id" in entry:
            reviewers[entry["reviewer_id"]].append(entry)

    results = []

    for reviewer_id, entries in reviewers.items():
        total_actions = len(entries)
        trust_deltas = [e.get("trust_delta", 0.0) for e in entries if "trust_delta" in e]
        helpful_count = sum(1 for delta in trust_deltas if delta > 0)

        avg_trust_delta = round(sum(trust_deltas) / len(trust_deltas), 4) if trust_deltas else 0.0
        helpful_pct = round((helpful_count / len(trust_deltas)) * 100, 2) if trust_deltas else 0.0

        results.append({
            "reviewer_id": reviewer_id,
            "total_actions": total_actions,
            "avg_trust_delta": avg_trust_delta,
            "helpful_override_pct": helpful_pct
        })

    return results

def write_scores(scores: List[dict], path: str):
    with open(path, "w") as f:
        for score in scores:
            f.write(json.dumps(score) + "\n")

def score_reviewers(input_path="logs/reviewer_impact_log.jsonl", output_path="logs/reviewer_scores.jsonl"):
    logs = load_logs(input_path)
    scores = compute_reviewer_scores(logs)
    write_scores(scores, output_path)

# Optional CLI test
if __name__ == "__main__":
    score_reviewers()
