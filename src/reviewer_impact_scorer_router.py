from fastapi import APIRouter
from pydantic import BaseModel
from src.utils.reviewer_scoring import score_reviewers
from pathlib import Path
import json
import os
import logging

router = APIRouter()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class ReviewerImpactEvent(BaseModel):
    reviewer_id: str
    trust_delta: float
    signal_id: str
    action: str
    note: str | None = None


@router.post("/internal/reviewer-impact-log")
def reviewer_impact_log(event: ReviewerImpactEvent):
    log_path = Path("logs/reviewer_impact_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        f.write(json.dumps(event.dict()) + "\n")

    logger.info(f"Impact log written to: {log_path}")
    logger.debug(f"Log entry: {event.dict()}")

    return {"status": "logged"}


@router.get("/internal/reviewer-scores")
def get_reviewer_scores():
    logger.debug("Starting reviewer scoring pipeline")

    try:
        files = os.listdir("logs")
        logger.debug(f"Files in logs/: {files}")
    except FileNotFoundError:
        logger.warning("logs/ directory not found")
        return {"error": "Logs directory not found."}

    score_reviewers()  # Runs the computation and writes the file

    output_path = "logs/reviewer_scores.jsonl"
    if not os.path.exists(output_path):
        logger.info("No reviewer scores file found after scoring")
        return {"error": "No reviewer scores file found."}

    with open(output_path) as f:
        results = [json.loads(line) for line in f if line.strip()]

    logger.debug(f"Loaded {len(results)} reviewer scores")
    return results