from src.schemas import ReviewerImpactLog
import logging

logger = logging.getLogger(__name__)

def log_reviewer_action(entry: ReviewerImpactLog) -> None:
    logger.info(f"Reviewer impact logged: {entry}")