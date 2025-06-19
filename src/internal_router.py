# src/internal_router.py

from fastapi import APIRouter
from pydantic import BaseModel
from collections import defaultdict
from typing import List
import json
import os
from src.cache_instance import cache

router = APIRouter(prefix="/internal", tags=["internal-tools"])

FEEDBACK_LOG_PATH = "data/feedback.jsonl"  # Adjust if your path differs

@router.get("/feedback-clusters")
def get_feedback_clusters():
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return {"clusters": []}

    clusters = defaultdict(list)

    with open(FEEDBACK_LOG_PATH, "r") as f:
        for line in f:
            try:
                feedback = json.loads(line)
                asset = feedback.get("asset", "unknown")
                clusters[asset].append(feedback)
            except json.JSONDecodeError:
                continue

    result = [{"asset": asset, "count": len(feeds), "samples": feeds[:3]} for asset, feeds in clusters.items()]
    return {"clusters": result}


# ✅ NEW: Inject mock signal directly into in-memory cache
class SignalEntry(BaseModel):
    score: float
    confidence: float
    label: str
    fallback_type: str = "mock"
    source: str = "twitter"
    trend: str = "upward"
    top_drivers: List[str] = ["mock"]
    price_at_score: float = 0.0
    type: str = "raw"

@router.post("/inject-test-signal/{asset}")
def inject_test_signal(asset: str, signal: SignalEntry):
    key = f"{asset.upper()}_history"
    existing = cache.get_signal(key) or []
    updated = existing + [signal.dict()]
    cache.set_signal(key, updated)
    return {
        "message": f"Injected test signal for {asset.upper()}",
        "total_signals": len(updated)
    }