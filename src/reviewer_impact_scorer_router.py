from fastapi import APIRouter
from pydantic import BaseModel
from src.utils.reviewer_scoring import score_reviewers
from pathlib import Path
import json
import os

router = APIRouter()


class ReviewerImpactEvent(BaseModel):
    reviewer_id: str
    trust_delta: float
    signal_id: str
    action: str


@router.post("/reviewer-impact-log")
def reviewer_impact_log(event: ReviewerImpactEvent):
    print("🚨 reviewer-impact-log route hit")

    log_path = Path("logs/reviewer_impact_log.jsonl")
    os.makedirs(log_path.parent, exist_ok=True)

    print(f"📄 Writing to: {log_path}")
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(event.dict()) + "\n")
    except Exception as e:
        print(f"❌ Write failed: {e}")

    return {"logged": True}


@router.get("/reviewer-scores")
def get_reviewer_scores():
    print("📊 reviewer-scores route hit")
    score_reviewers()  # Runs the computation and writes the file

    output_path = "logs/reviewer_scores.jsonl"
    if not os.path.exists(output_path):
        print(f"⚠️ reviewer_scores.jsonl not found at: {output_path}")
        return {"error": "No reviewer scores file found."}

    with open(output_path) as f:
        return [json.loads(line) for line in f if line.strip()]