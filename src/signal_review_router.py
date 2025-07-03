# src/signal_review_router.py

from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
import json
import os

router = APIRouter(prefix="/internal", tags=["signal-review"])

SUPPRESSION_REVIEW_PATH = "data/suppression_review_queue.jsonl"
RETRAIN_QUEUE_PATH = "data/retrain_queue.jsonl"
OVERRIDE_LOG_PATH = "data/override_log.jsonl"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, entry):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


@router.get("/review-suppressed")
def review_suppressed_signals():
    review_data = load_jsonl(SUPPRESSION_REVIEW_PATH)
    return {"review_queue": review_data}


@router.post("/flag-for-retraining")
async def flag_signal_for_retraining(request: Request):
    body = await request.json()
    signal_id = body.get("signal_id")
    reason = body.get("reason", "unspecified")
    note = body.get("note", "")

    review_queue = load_jsonl(SUPPRESSION_REVIEW_PATH)
    matching_signal = next((s for s in review_queue if s["id"] == signal_id), None)

    if not matching_signal:
        raise HTTPException(status_code=404, detail="Signal not found in suppression queue")

    # Check for duplicates
    retrain_queue = load_jsonl(RETRAIN_QUEUE_PATH)
    if any(s["id"] == signal_id for s in retrain_queue):
        return {"status": "ok", "added": False, "signal_id": signal_id, "reason": "already exists"}

    matching_signal.update({
        "flagged_for_retraining": True,
        "flag_reason": reason,
        "flagged_at": datetime.utcnow().isoformat(),
    })

    if note:
        matching_signal["note"] = note

    append_jsonl(RETRAIN_QUEUE_PATH, matching_signal)

    return {"status": "ok", "added": True, "signal_id": signal_id}


@router.post("/override-suppression")
async def override_suppressed_signal(request: Request):
    body = await request.json()
    signal_id = body.get("signal_id")
    override_reason = body.get("override_reason", "unspecified")
    note = body.get("note", "")
    reviewed_by = "founder"

    review_queue = load_jsonl(SUPPRESSION_REVIEW_PATH)
    signal = next((s for s in review_queue if s["id"] == signal_id), None)

    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found in suppression queue")

    override_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "signal_id": signal_id,
        "override_reason": override_reason,
        "source": "manual_override",
        "reviewed_by": reviewed_by,
        "full_payload": signal,
    }

    if note:
        override_entry["note"] = note

    append_jsonl(OVERRIDE_LOG_PATH, override_entry)

    return {"status": "ok", "override_applied": True, "signal_id": signal_id}
