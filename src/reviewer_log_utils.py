from src.schemas import ReviewerImpactLog
import logging

logger = logging.getLogger(__name__)

def log_reviewer_action(signal_id: str, reviewer_id: str, action: str, note: str = ""):
    # Example logging logic — replace this with actual DB write, etc.
    print(f"[REVIEWER LOG] signal_id={signal_id}, reviewer_id={reviewer_id}, action={action}, note={note}")
