from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import logging

from src.signal_log import log_signal  # ✅ NEW

router = APIRouter()
logging.basicConfig(level=logging.INFO)

class Feedback(BaseModel):
    asset: str
    sentiment: float
    user_feedback: str
    timestamp: str  # Expecting ISO string from frontend
    context: str | None = None  # Optional field

@router.options("/feedback")
async def options_feedback():
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )

@router.post("/feedback")
async def receive_feedback(feedback: Feedback):
    log_data = {
        "type": "user_feedback",
        "timestamp": datetime.utcnow().isoformat(),
        "asset": feedback.asset,
        "sentiment": feedback.sentiment,
        "user_feedback": feedback.user_feedback,
        "context": feedback.context,
        "source": "frontend"
    }

    logging.info({
        "event": "user_feedback_received",
        "timestamp": log_data["timestamp"],
        "data": feedback.dict()
    })

    # ✅ Log feedback to signal_history.jsonl
    log_signal(log_data)

    return {"status": "received", "received_at": log_data["timestamp"]}
