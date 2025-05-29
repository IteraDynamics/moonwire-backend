from fastapi import APIRouter, Body
from datetime import datetime

router = APIRouter()

# Temporary in-memory store
feedback_store = []

@router.post("/feedback")
def submit_feedback(
    signal_id: str = Body(...),
    asset: str = Body(...),
    score: float = Body(...),
    user_confidence: str = Body(...),
    user_agrees: bool = Body(...),
    comments: str = Body(""),
):
    feedback_entry = {
        "signal_id": signal_id,
        "asset": asset,
        "score": score,
        "user_confidence": user_confidence,
        "user_agrees": user_agrees,
        "comments": comments,
        "timestamp": datetime.utcnow().isoformat()
    }
    feedback_store.append(feedback_entry)
    return {"status": "received", "entry": feedback_entry}