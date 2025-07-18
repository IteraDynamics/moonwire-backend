from fastapi import APIRouter
from src.utils.reviewer_scoring import score_reviewers
import json
import os

router = APIRouter()

@router.get("/internal/reviewer-scores")
def get_reviewer_scores():
    score_reviewers()  # Runs the computation and writes the file

    output_path = "logs/reviewer_scores.jsonl"
    if not os.path.exists(output_path):
        return {"error": "No reviewer scores file found."}

    with open(output_path) as f:
        return [json.loads(line) for line in f if line.strip()]
