from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import json
import uuid

router = APIRouter()
FEEDBACK_LOG = Path("data/feedback.jsonl")
SIGNAL_LOG = Path("logs/signal_history.jsonl")
RETRAIN_QUEUE = Path("data/retrain_queue.jsonl")

# Ensure dirs exist
FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
RETRAIN_QUEUE.parent.mkdir(parents=True, exist_ok=True)

class FeedbackEntry(BaseModel):
    signal_id: str
    user_id: str
    agree: bool
    timestamp: str
    note: str | None = None

@router.post("/feedback")
def submit_feedback(entry: FeedbackEntry):
    # Load signal log to validate signal_id
    if not SIGNAL_LOG.exists():
        raise HTTPException(status_code=500, detail="Signal history file missing")

    with open(SIGNAL_LOG, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    matching_signal = next((s for s in lines if s.get("id") == entry.signal_id), None)
    if not matching_signal:
        raise HTTPException(status_code=404, detail="Signal ID not found")

    feedback_record = {
        "id": f"fb_{uuid.uuid4().hex[:8]}",
        "signal_id": entry.signal_id,
        "user_id": entry.user_id,
        "agree": entry.agree,
        "timestamp": entry.timestamp,
        "note": entry.note
    }

    # Append feedback to feedback.jsonl
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(feedback_record) + "\n")

    # Update signal_history.jsonl to include feedback_refs
    for s in lines:
        if s.get("id") == entry.signal_id:
            refs = s.get("feedback_refs", [])
            refs.append(feedback_record["id"])
            s["feedback_refs"] = refs
            s["feedback_updated_at"] = datetime.utcnow().isoformat()

    # Rewrite full log with updated signal
    with open(SIGNAL_LOG, "w") as f:
        for s in lines:
            f.write(json.dumps(s) + "\n")

    # If disagree, add to retrain_queue.jsonl
    if entry.agree is False:
        with open(RETRAIN_QUEUE, "a") as f:
            f.write(json.dumps({
                "signal_id": entry.signal_id,
                "feedback_id": feedback_record["id"],
                "user_id": entry.user_id,
                "timestamp": entry.timestamp
            }) + "\n")

    return {"status": "recorded", "feedback_id": feedback_record["id"]}
