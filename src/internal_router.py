# src/internal_router.py

from fastapi import APIRouter
from collections import defaultdict
import json
import os

router = APIRouter()

FEEDBACK_LOG_PATH = "data/feedback.jsonl"  # Adjust if your path differs

@router.get("/internal/feedback-clusters")
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
