from fastapi import APIRouter, Query
from src.signal_utils import generate_composite_signal, compute_trust_scores
from src.feedback_utils import get_feedback_summary_for_signal, run_disagreement_prediction

router = APIRouter()

@router.get("/composite")
def get_composite_signal(asset: str = Query(...), twitter_score: float = Query(...), news_score: float = Query(...)):
    """
    Generate a composite signal based on sentiment scores and trust insights.
    """
    signal = generate_composite_signal(asset, twitter_score, news_score)
    
    # Fetch trust data
    trust_insights = {}
    feedback_summary = get_feedback_summary_for_signal(signal["id"])

    confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
    predicted_disagreement_prob = run_disagreement_prediction(
        score=signal["score"],
        confidence=confidence_map[signal["confidence"]],
        label=signal["label"]
    )

    trust_insights[signal["id"]] = {
        "historical_agreement_rate": feedback_summary.get("historical_agreement_rate"),
        "predicted_disagreement_prob": predicted_disagreement_prob
    }

    compute_trust_scores(signal, trust_insights)
    return signal