# src/consensus_dashboard_router.py

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter

router = APIRouter(prefix="/internal")

CONSENSUS_THRESHOLD = 2.5
DAYS_BACK = 7


def _score_to_weight(score: Optional[float]) -> float:
    if score is None:
        return 1.0
    if score >= 0.75:
        return 1.25
    if score >= 0.5:
        return 1.0
    return 0.75


@router.get("/consensus-dashboard")
def consensus_dashboard():
    """
    Live summary of signals flagged for retraining in the last N days,
    with deduped reviewers, banded fallback weights, total_weight, and trigger flag.
    """
    # Import inside to honor per-test reload of src.paths
    import src.paths as paths

    retraining_path = Path(paths.RETRAINING_LOG_PATH)
    scores_path = Path(paths.REVIEWER_SCORES_PATH)

    # Load fallback scores (raw)
    raw_scores: Dict[str, float] = {}
    if scores_path.exists():
        with scores_path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = rec.get("reviewer_id")
                if rid:
                    raw_scores[rid] = rec.get("score")

    # If no retraining log, return empty list
    if not retraining_path.exists():
        return []

    cutoff = datetime.utcnow().timestamp() - DAYS_BACK * 24 * 3600

    # Aggregate by signal
    signals: Dict[str, Dict] = {}
    with retraining_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = entry.get("timestamp")
            if not isinstance(ts, (int, float)):
                ts = datetime.utcnow().timestamp()
            if ts < cutoff:
                continue

            sig = entry.get("signal_id")
            rid = entry.get("reviewer_id")
            if not sig or not rid:
                continue

            bucket = signals.setdefault(sig, {
                "signal_id": sig,
                "reviewers": [],      # [{"id","weight"}]
                "seen": set(),
                "total_weight": 0.0,
                "last_flagged_ts": ts
            })

            if ts > bucket["last_flagged_ts"]:
                bucket["last_flagged_ts"] = ts

            if rid in bucket["seen"]:
                continue  # first flag from reviewer counts

            weight = entry.get("reviewer_weight")
            if weight is None:
                weight = _score_to_weight(raw_scores.get(rid))

            bucket["seen"].add(rid)
            bucket["reviewers"].append({"id": rid, "weight": weight})
            bucket["total_weight"] += weight

    results: List[Dict] = []
    for bucket in signals.values():
        results.append({
            "signal_id": bucket["signal_id"],
            "reviewers": bucket["reviewers"],
            "total_weight": bucket["total_weight"],
            "triggered": bucket["total_weight"] >= CONSENSUS_THRESHOLD,
            "last_flagged_timestamp": datetime.utcfromtimestamp(bucket["last_flagged_ts"]).isoformat() + "Z",
        })

    # Sort by total_weight desc, then signal_id for stability
    results.sort(key=lambda x: (-x["total_weight"], x["signal_id"]))
    return results