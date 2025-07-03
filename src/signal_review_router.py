from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import json
import os

router = APIRouter(prefix="/internal", tags=["signal-review"])

SUPPRESSION_QUEUE_PATH = "data/suppression_review_queue.jsonl"
RETRAIN_QUEUE_PATH = "data/retrain_queue.jsonl"


class RetrainingRequest(BaseModel):
    signal_id: str
    reason: str
    note: str = None


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_entry(path, entry):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


@router.post("/flag-for-retraining")
def flag_signal_for_retraining(payload: RetrainingRequest):
    # Load suppression review queue
    suppressed_signals = load_jsonl(SUPPRESSION_QUEUE_PATH)
    match = next((s for s in suppressed_signals if s.get("id") == payload.signal_id), None)

    if not match:
        raise HTTPException(status_code=404, detail="Signal not found in suppression queue")

    # Load retrain queue to prevent duplicates
    existing = load_jsonl(RETRAIN_QUEUE_PATH)
    if any(entry.get("id") == payload.signal_id for entry in existing):
        return {"status": "ok", "added": False, "reason": "already_exists", "signal_id": payload.signal_id}

    retrain_entry = {
        **match.get("full_payload", match),
        "flagged_for_retraining": True,
        "flag_reason": payload.reason,
        "flagged_at": datetime.utcnow().isoformat(),
    }

    if payload.note:
        retrain_entry["note"] = payload.note

    write_jsonl_entry(RETRAIN_QUEUE_PATH, retrain_entry)

    return {"status": "ok", "added": True, "signal_id": payload.signal_id}
