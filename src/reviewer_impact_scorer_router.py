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

@router.post("/internal/reviewer-impact-log")
def reviewer_impact_log(event: ReviewerImpactEvent):
    log_path = Path("logs/reviewer_impact_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        f.write(json.dumps(event.dict()) + "\n")
        f.flush()        # ensure it's written to buffer
        os.fsync(f.fileno())  # force flush to disk

    return {"status": "logged"}

@router.get("/internal/reviewer-scores")
def get_reviewer_scores():
    print("[DEBUG] Calling score_reviewers()")
    score_reviewers()

    output_path = "logs/reviewer_scores.jsonl"
    if not os.path.exists(output_path):
        return {"error": "No reviewer scores file found."}

    with open(output_path) as f:
        return [json.loads(line) for line in f if line.strip()]