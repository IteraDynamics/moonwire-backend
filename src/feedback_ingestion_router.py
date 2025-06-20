from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path

router = APIRouter()

FEEDBACK_LOG = Path("data/feedback.jsonl")
SIGNAL_LOG = Path("logs/signal_history.jsonl")
RETRAIN_QUEUE = Path("data/retrain_queue.jsonl")


class FeedbackEntry(BaseModel):
    signal_id: str
    user_id: str
    agree: bool
    timestamp: str
    note: Optional[str] = None


@router.post("/internal/feedback")
def receive_feedback(entry: FeedbackEntry):
    # ✅ Ensure signal_id exists in signal_history.jsonl
    if not SIGNAL_LOG.exists():
        raise HTTPException(status_code=400, detail="No signal history found")

    matching = []
    with open(SIGNAL_LOG, "r") as f:
        for line in f:
            row = json.loads(line)
            if row.get("id") == entry.signal_id:
                matching.append(row)

    if not matching:
        raise HTTPException(status_code=404, detail="Signal ID not found")

    # ✅ Write to feedback.jsonl
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(entry.dict()) + "\n")

    # ✅ Append to retrain_queue.jsonl if user disagreed
    if entry.agree is False:
        with open(RETRAIN_QUEUE, "a") as f:
            f.write(json.dumps(entry.dict()) + "\n")

    return {"status": "ok", "message": "Feedback received"}