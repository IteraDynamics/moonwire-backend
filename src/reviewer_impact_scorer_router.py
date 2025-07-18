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
    log_path = Path("logs/reviewer_impact_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        f.write(json.dumps(event.dict()) + "\n")

    return {"logged": True}

@router.get("/reviewer-scores")
def get_reviewer_scores():
    score_reviewers()  # Runs the computation and writes the file

    output_path = "logs/reviewer_scores.jsonl"
    if not os.path.exists(output_path):
        return {"error": "No reviewer scores file found."}

    with open(output_path) as f:
        return [json.loads(line) for line in f if line.strip()]

@router.get("/dev-dump-reviewer-logs")
def dev_dump_reviewer_logs():
    log_path = Path("logs/reviewer_impact_log.jsonl")
    
    if not log_path.exists():
        return {"error": "Log file does not exist."}
    
    with open(log_path, "r") as f:
        return {"log_contents": f.read()}