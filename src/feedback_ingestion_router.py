# src/feedback_ingestion_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from pathlib import Path
import json

router = APIRouter()
FEEDBACK_PATH = Path("data/feedback.jsonl")
RETRAIN_QUEUE = Path("data/retrain_queue.jsonl")
SIGNAL_LOG = Path("logs/signal_history.jsonl")

class FeedbackEntry(BaseModel):
    asset: str
    signal_id: str
    user_id: str
    agree: bool
    timestamp: str
    note: Optional[str] = None

@router.post("/feedback")
def receive_feedback(entry: FeedbackEntry):
    # ✅ Validate signal ID exists
    if not SIGNAL_LOG.exists():
        raise HTTPException(status_code=400, detail="Signal log not found.")

    with open(SIGNAL_LOG, "r") as f:
        signal_exists = any(entry.signal_id in line for line in f)

    if not signal_exists:
        raise HTTPException(status_code=404, detail="Signal ID not found.")

    # ✅ Write feedback to log
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(entry.dict()) + "\n")

    # ✅ If disagreement, add to retrain queue
    if entry.agree is False:
        RETRAIN_QUEUE.parent.mkdir(parents=True, exist_ok=True)
        with open(RETRAIN_QUEUE, "a") as f:
            f.write(json.dumps(entry.dict()) + "\n")

    return {
        "message": "Feedback recorded",
        "signal_id": entry.signal_id,
        "agree": entry.agree
    }