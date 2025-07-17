from pydantic import BaseModel

class ReviewerImpactLog(BaseModel):
    reviewer_id: str
    impact_score: float
    comment: str | None = None  # Optional field