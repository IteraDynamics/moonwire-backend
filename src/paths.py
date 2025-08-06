# src/paths.py
import os
from pathlib import Path

# Root directory where all logs live
LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))

# Ensure the directory exists when imported
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Path to the reviewer impact log (override & suppression logs)
REVIEWER_IMPACT_LOG_PATH = LOGS_DIR / "reviewer_impact_log.jsonl"

# Path to the reviewer scores (trust‐weight per reviewer)
REVIEWER_SCORES_PATH = LOGS_DIR / "reviewer_scores.jsonl"

# NEW: path to retraining requests log
RETRAINING_LOG_PATH = LOGS_DIR / "retraining_log.jsonl"