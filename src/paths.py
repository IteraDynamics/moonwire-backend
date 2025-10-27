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

# Historical time-series for reviewer scores
REVIEWER_SCORES_HISTORY_PATH = LOGS_DIR / "reviewer_scores_history.jsonl"

# Shadow inference and signal logs
SHADOW_LOG_PATH = LOGS_DIR / "signal_inference_shadow.jsonl"
SIGNAL_HISTORY_LOG_PATH = LOGS_DIR / "signal_history.jsonl"

# Feedback and review queues
FEEDBACK_LOG_PATH = LOGS_DIR / "feedback.jsonl"
SUPPRESSION_REVIEW_PATH = LOGS_DIR / "suppression_review_queue.jsonl"
RETRAIN_QUEUE_PATH = LOGS_DIR / "retrain_queue.jsonl"

# Governance parameters
GOVERNANCE_PARAMS_PATH = MODELS_DIR / "governance_params.json"

# ML hyperparameters (training-time config)
ML_HYPERPARAMETERS_PATH = MODELS_DIR / "ml_hyperparameters.json"

# Paper trading parameters (simulation config)
PAPER_TRADING_PARAMS_PATH = MODELS_DIR / "paper_trading_params.json"

# Performance metrics
PERFORMANCE_METRICS_PATH = MODELS_DIR / "performance_metrics.json"

# Note: TRIGGER_LOG_PATH and TRAINING_VERSION_FILE are defined in src/ml/infer.py
# as module-level variables to support test env var overrides via importlib.reload()