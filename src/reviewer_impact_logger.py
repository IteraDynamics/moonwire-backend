import os
import json
from datetime import datetime

REVIEWER_IMPACT_LOG_PATH = "data/reviewer_impact_log.jsonl"
os.makedirs("data", exist_ok=True)  # Ensure data directory exists

def log_reviewer_impact(
    reviewer_id,
    signal_id,
    action_type,  # should be either "overridden" or "retrained"
    original_trust_score,
    trust_score_before,
    trust_score_after,
    signal_timestamp,
    reviewer_note,
    reason,
    model_version="v1.0"
):
    log_entry = {
        "reviewer_id": reviewer_id,
        "signal_id": signal_id,
        "action_type": action_type,
        "original_trust_score": original_trust_score,
        "trust_score_before": trust_score_before,
        "trust_score_after": trust_score_after,
        "signal_timestamp": signal_timestamp,
        "reviewer_note": reviewer_note,
        "model_version": model_version,
        "logged_at": datetime.utcnow().isoformat()
    }

    # Assign override/retrain reason explicitly
    reason_key = "override_reason" if action_type == "overridden" else "retrain_reason"
    log_entry[reason_key] = reason

    try:
        with open(REVIEWER_IMPACT_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"❌ Failed to log reviewer impact: {e}")
        raise
