from fastapi import APIRouter
from src.signal_utils import generate_composite_signal, compute_trust_scores
from src.feedback_utils import get_feedback_summary_for_signal
from src.twitter_signal import fetch_disagreement_prediction

router = APIRouter()


@router.get("/composite-signal")
async def get_composite_signal():
    composite_signal = generate_composite_signal()
    trust_insights = compute_trust_scores(fetch_disagreement_prediction())

    trust_lookup = {entry["signal_id"]: entry for entry in trust_insights}
    for signal in composite_signal["signals"]:
        trust_info = trust_lookup.get(signal["id"])
        if trust_info:
            signal["trust_score"] = trust_info["trust_score"]
            signal["trust_label"] = trust_info["trust_label"]

    return composite_signal


@router.get("/composite-signal/feedback-summary")
async def get_feedback_summary():
    return get_feedback_summary_for_signal()