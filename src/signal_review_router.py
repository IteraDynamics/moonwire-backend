from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import os
from pathlib import Path
from collections import defaultdict

router = APIRouter(prefix="/internal", tags=["internal-tools"])

SUPPRESSION_REVIEW_PATH = Path("data/suppression_review_queue.jsonl")
RETRAIN_QUEUE_PATH = Path("data/retrain_queue.jsonl")
OVERRIDE_LOG_PATH = Path("data/override_log.jsonl")


class RetrainRequest(BaseModel):
    signal_id: str
    reason: str
    note: Optional[str] = None


class OverrideRequest(BaseModel):
    signal_id: str
    override_reason: str
    note: Optional[str] = None
    reviewed_by: Optional[str] = "founder"


@router.post("/flag-for-retraining")
def flag_for_retraining(req: RetrainRequest):
    if not SUPPRESSION_REVIEW_PATH.exists():
        raise HTTPException(status_code=404, detail="Suppression review file not found.")

    matched_signal = None
    with SUPPRESSION_REVIEW_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("id") == req.signal_id:
                    matched_signal = entry
                    break
            except json.JSONDecodeError:
                continue

    if not matched_signal:
        raise HTTPException(status_code=404, detail="Signal not found in review queue.")

    new_entry = {
        **matched_signal.get("full_payload", {}),
        "flagged_for_retraining": True,
        "flag_reason": req.reason,
        "note": req.note,
        "flagged_at": datetime.utcnow().isoformat()
    }

    os.makedirs(RETRAIN_QUEUE_PATH.parent, exist_ok=True)
    with RETRAIN_QUEUE_PATH.open("a") as f:
        f.write(json.dumps(new_entry) + "\n")

    return {"status": "ok", "added": True, "signal_id": req.signal_id}


@router.post("/override-suppression")
def override_suppressed_signal(req: OverrideRequest):
    if not SUPPRESSION_REVIEW_PATH.exists():
        raise HTTPException(status_code=404, detail="Suppression review file not found.")

    matched_signal = None
    with SUPPRESSION_REVIEW_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("id") == req.signal_id:
                    matched_signal = entry
                    break
            except json.JSONDecodeError:
                continue

    if not matched_signal:
        raise HTTPException(status_code=404, detail="Signal not found in review queue.")

    override_log = {
        "timestamp": datetime.utcnow().isoformat(),
        "signal_id": req.signal_id,
        "override_reason": req.override_reason,
        "source": "manual_override",
        "reviewed_by": req.reviewed_by,
        "note": req.note,
        "full_payload": matched_signal.get("full_payload", {})
    }

    os.makedirs(OVERRIDE_LOG_PATH.parent, exist_ok=True)
    with OVERRIDE_LOG_PATH.open("a") as f:
        f.write(json.dumps(override_log) + "\n")

    return {"status": "ok", "override_applied": True, "signal_id": req.signal_id}


@router.get("/retrain-summary")
def retrain_summary():
    if not RETRAIN_QUEUE_PATH.exists():
        return {
            "total_flagged": 0,
            "assets": {},
            "flag_reasons": {},
            "latest_entries": []
        }

    entries = []
    with RETRAIN_QUEUE_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                continue

    total_flagged = len(entries)
    assets = defaultdict(int)
    flag_reasons = defaultdict(int)

    for entry in entries:
        assets[entry.get("asset", "Unknown")] += 1
        flag_reasons[entry.get("flag_reason", "Unknown")] += 1

    sorted_entries = sorted(entries, key=lambda e: e.get("flagged_at", ""), reverse=True)
    latest_entries = []
    for entry in sorted_entries[:10]:
        latest_entries.append({
            "signal_id": entry.get("id"),
            "flag_reason": entry.get("flag_reason"),
            "trust_score": entry.get("trust_score", "Unknown"),
            "flagged_at": entry.get("flagged_at", "Unknown")
        })

    return {
        "total_flagged": total_flagged,
        "assets": dict(assets),
        "flag_reasons": dict(flag_reasons),
        "latest_entries": latest_entries
    }
