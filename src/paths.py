# src/paths.py
from pathlib import Path
import os

# Project root (…/moonwire-backend)
BASE_DIR = Path(__file__).resolve().parent.parent

# Allow tests (and CI) to override via env vars
LOGS_DIR = Path(os.getenv("LOGS_DIR", str(BASE_DIR / "logs"))).resolve()
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models"))).resolve()  # <— NEW
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(BASE_DIR / "artifacts"))).resolve()

# Ensure these exist at import time (tests may write directly into them)
for _d in (LOGS_DIR, MODELS_DIR, ARTIFACTS_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# Existing logs
REVIEWER_IMPACT_LOG_PATH = LOGS_DIR / "reviewer_impact_log.jsonl"
REVIEWER_SCORES_PATH = LOGS_DIR / "reviewer_scores.jsonl"

# Consensus / retraining logs
RETRAINING_LOG_PATH = LOGS_DIR / "retraining_log.jsonl"
RETRAINING_TRIGGERED_LOG_PATH = LOGS_DIR / "retraining_triggered.jsonl"

# NEW: historical time-series for reviewer scores
REVIEWER_SCORES_HISTORY_PATH = LOGS_DIR / "reviewer_scores_history.jsonl"