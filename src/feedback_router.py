# src/feedback_router.py

from fastapi import APIRouter, Request
from pydantic import BaseModel
from datetime import datetime
import logging

router = APIRouter()

logging.basicConfig(level=logging.INFO)

class Feedback(BaseModel):
    asset: str
    sentiment: float
    user_feedback: str
    timestamp: str  # Expecting ISO string from frontend
    context: str | None = None  # Optional field

@router.post("/feedback")
async def receive_feedback(feedback: Feedback):
    logging.info({
        "event": "user_feedback_received",
        "timestamp": datetime.utcnow().isoformat(),
        "data": feedback.dict()
    })

    return {"status": "received", "received_at": datetime.utcnow().isoformat()}