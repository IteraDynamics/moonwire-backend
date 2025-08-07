# src/consensus_dashboard_router.py

import json
from typing import Dict, List
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import APIRouter
from src.paths import RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH

router = APIRouter()

RETRAINING_THRESHOLD = 2.5

@router.get("/internal/consensus-dashboard")
def consensus_dashboard():
    now = datetime.utcnow()
    seven_days_ago = now - timedelta(days=7)

    # Load reviewer scores for fallback
    scores = {}
    scores_path = Path(REVIEWER_SCORES_PATH)
    if scores_path.exists():
        with open(scores_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                scores[entry["reviewer_id"]] = entry.get("score", 1.0)

    # Parse retraining logs
    signals: Dict[str, Dict] = {}
    log_path = Path(RETRAINING_LOG_PATH)
    if log_path.exists():
        with open(log_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                ts = datetime.utcfromtimestamp(entry["timestamp"])
                if ts < seven_days_ago:
                    continue

                sid = entry["signal_id"]
                rid = entry["reviewer_id"]
                weight = entry.get("reviewer_weight", None)

                if sid not in signals:
                    signals[sid] = {
                        "signal_id": sid,
                        "reviewers": {},
                        "last_flagged_timestamp": ts,
                    }

                if rid not in signals[sid]["reviewers"]:
                    w = weight if weight is not None else scores.get(rid, 1.0)
                    signals[sid]["reviewers"][rid] = w

                # Update last flagged time
                if ts > signals[sid]["last_flagged_timestamp"]:
                    signals[sid]["last_flagged_timestamp"] = ts

    results = []
    for s in signals.values():
        reviewers = [{"id": rid, "weight": w} for rid, w in s["reviewers"].items()]
        total = sum(w["weight"] for w in reviewers)
        results.append({
            "signal_id": s["signal_id"],
            "reviewers": reviewers,
            "total_weight": total,
            "triggered": total >= RETRAINING_THRESHOLD,
            "last_flagged_timestamp": s["last_flagged_timestamp"].isoformat(),
        })

    results.sort(key=lambda x: x["total_weight"], reverse=True)
    return results