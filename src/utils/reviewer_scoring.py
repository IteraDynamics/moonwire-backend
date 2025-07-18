# src/utils/reviewer_scoring.py

import json
from collections import defaultdict
from typing import List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Resolve absolute path to repo root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_PATH = BASE_DIR / "logs" / "reviewer_impact_log.jsonl"
DEFAULT_OUTPUT_PATH = BASE_DIR / "logs" / "reviewer_scores.jsonl"

def load_logs(path: Path) -> List[dict]:
    logger.debug(f"Loading logs from: {path}")
    if not path.exists():
        logger.debug("Log file does not exist.")
        return []
    with path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]
    logger.debug(f"Loaded {len(lines)} log lines")
    return [json.loads(line) for line in lines]

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

def write_scores(scores: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for score in scores:
            f.write(json.dumps(score) + "\n")
    logger.debug(f"Wrote {len(scores)} scores to: {path}")

def score_reviewers(input_path: Path = DEFAULT_INPUT_PATH, output_path: Path = DEFAULT_OUTPUT_PATH):
    logger.debug("[DEBUG] Starting reviewer scoring pipeline")
    logs = load_logs(input_path)
    if not logs:
        logger.info("No logs found or log file is empty — skipping scoring")
        return
    scores = compute_reviewer_scores(logs)
    write_scores(scores, output_path)