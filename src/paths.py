# src/paths.py
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# Allow tests (and CI) to override via env vars
LOGS_DIR = Path(os.getenv("LOGS_DIR", str(BASE_DIR / "logs")))
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))  # <— NEW

# Existing logs
REVIEWER_IMPACT_LOG_PATH = LOGS_DIR / "reviewer_impact_log.jsonl"
REVIEWER_SCORES_PATH = LOGS_DIR / "reviewer_scores.jsonl"

# Consensus / retraining logs
RETRAINING_LOG_PATH = LOGS_DIR / "retraining_log.jsonl"
RETRAINING_TRIGGERED_LOG_PATH = LOGS_DIR / "retraining_triggered.jsonl"

# NEW: historical time-series for reviewer scores
REVIEWER_SCORES_HISTORY_PATH = LOGS_DIR / "reviewer_scores_history.jsonl"
