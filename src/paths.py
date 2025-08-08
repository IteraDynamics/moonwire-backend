# src/paths.py

import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Allow tests (and prod) to override logs directory via env var.
# Falls back to the repo's logs/ directory.
DEFAULT_LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR = Path(os.getenv("LOGS_DIR", str(DEFAULT_LOGS_DIR)))

# Core logs used across features
REVIEWER_IMPACT_LOG_PATH = LOGS_DIR / "reviewer_impact_log.jsonl"
REVIEWER_SCORES_PATH = LOGS_DIR / "reviewer_scores.jsonl"
RETRAINING_LOG_PATH = LOGS_DIR / "retraining_log.jsonl"
RETRAINING_TRIGGERED_LOG_PATH = LOGS_DIR / "retraining_triggered.jsonl"