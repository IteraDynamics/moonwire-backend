import os
import json
from datetime import datetime

REVIEWER_IMPACT_LOG_PATH = "data/reviewer_impact_log.jsonl"

def log_reviewer_impact(
    reviewer_id: str,
    signal_id: str,
    action_type: str,  # "override" or "retrain_flag"
    original_trust_score: float,
    signal_timestamp: str,
    reviewer_note: str = None,
    reason: str = None,
    model_version: str = "v2.3"
):
    """
    Logs reviewer intervention metadata to reviewer_impact_log.jsonl.
    """
    entry = {
        "reviewer_id": reviewer_id,
        "signal_id": signal_id,
        "action_type": action_type,
        "original_trust_score": original_trust_score,
        "signal_timestamp": signal_timestamp,
        "reviewer_note": reviewer_note,
        "override_reason" if action_type == "override" else "retrain_reason": reason,
        "current_model_version": model_version,
        "logged_at": datetime.utcnow().isoformat()
    }

    os.makedirs(os.path.dirname(REVIEWER_IMPACT_LOG_PATH), exist_ok=True)
    with open(REVIEWER_IMPACT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
