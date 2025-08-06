from pathlib import Path

# Where all our JSONL logs live
LOGS_DIR = Path("logs")

# Impact and scoring logs
REVIEWER_IMPACT_LOG_PATH = LOGS_DIR / "reviewer_impact_log.jsonl"
REVIEWER_SCORES_PATH     = LOGS_DIR / "reviewer_scores.jsonl"

# Retraining (flag-for-retraining) log
RETRAINING_LOG_PATH      = LOGS_DIR / "retraining_log.jsonl"