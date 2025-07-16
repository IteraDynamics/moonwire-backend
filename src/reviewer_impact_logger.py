import os
import json
from datetime import datetime
from src.schemas import ReviewerImpactLog

REVIEWER_IMPACT_LOG_PATH = "data/reviewer_impact_log.jsonl"
os.makedirs("data", exist_ok=True)  # Ensure data directory exists

def log_reviewer_impact(log: ReviewerImpactLog):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "signal_id": log.signal_id,
        "reviewer_id": log.reviewer_id,
        "action": log.action,
        "note": log.note
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
