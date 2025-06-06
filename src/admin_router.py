# src/admin_router.py

from fastapi import APIRouter
from datetime import datetime, timedelta
from typing import List
from pydantic import BaseModel

router = APIRouter(prefix="/admin", tags=["admin"])

# === Response Schema ===
class FeedbackStats(BaseModel):
    asset: str
    feedback_count: int
    avg_score_delta: float
    user_agrees_rate: float
    avg_user_confidence: float
    last_feedback_time: str  # ISO timestamp string

# === Mock Data (replace with real aggregation logic later) ===
@router.get("/feedback-stats", response_model=List[FeedbackStats])
async def get_feedback_stats():
    return [
        FeedbackStats(
            asset="BTC",
            feedback_count=12,
            avg_score_delta=-0.2,
            user_agrees_rate=0.67,
            avg_user_confidence=0.81,
            last_feedback_time=(datetime.utcnow() - timedelta(hours=2)).isoformat()
        ),
        FeedbackStats(
            asset="ETH",
            feedback_count=7,
            avg_score_delta=0.1,
            user_agrees_rate=0.57,
            avg_user_confidence=0.72,
            last_feedback_time=(datetime.utcnow() - timedelta(hours=1)).isoformat()
        )
    ]
