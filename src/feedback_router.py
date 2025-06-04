# src/feedback_router.py

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
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
    logging.info({
        "event": "user_feedback_received",
        "timestamp": datetime.utcnow().isoformat(),
        "data": feedback.dict()
    })
    return {"status": "received", "received_at": datetime.utcnow().isoformat()}