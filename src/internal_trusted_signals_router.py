# src/internal_trusted_signals_router.py

from fastapi import APIRouter
from src.signal_cache import get_latest_signal
from src.feedback_utils import get_feedback_summary_for_signal, get_disagreement_probability
from src.signal_utils import compute_trust_scores

router = APIRouter(prefix="/internal")

@router.get("/signals-with-trust")
def get_signals_with_trust():
    """
    Returns the latest signal enriched with trust metadata.
    If no signal is cached yet, returns a 404-style message.
    """
    signal = get_latest_signal()

    if not signal:
        return {"error": "No signal available in cache."}

    signal_id = signal.get("id")
    feedback = get_feedback_summary_for_signal(signal_id)
    disagreement_prob = get_disagreement_probability(
        label=signal["label"],
        score=signal["score"],
        confidence=signal["confidence"]
    )

    trust_data = {
        signal_id: {
            "historical_agreement_rate": feedback["historical_agreement_rate"],
            "predicted_disagreement_prob": disagreement_prob
        }
    }

    compute_trust_scores(signal, trust_data)

    enriched = {
        **signal,
        "trust_score": signal.get("trust_score"),
        "trust_label": signal.get("trust_label"),
        "likely_disagreed": disagreement_prob > 0.5,
        "num_feedback": feedback["num_feedback"],
        "num_agree": feedback["num_agree"],
        "num_disagree": feedback["num_disagree"],
        "historical_agreement_rate": feedback["historical_agreement_rate"]
    }

    return enriched
    
@router.post("/inject-mock-signal")
def inject_mock_signal():
    """
    TEMPORARY:
    Injects a mock signal into the in-memory cache for testing.
    """
    from src.signal_cache import signal_cache

    mock_signal = {
        "id": "mock-signal-001",
        "label": "Positive",
        "score": 0.85,
        "confidence": 0.6,
        "timestamp": "2025-06-26T09:30:00Z"
    }

    signal_cache["latest"] = mock_signal
    return {"status": "mock signal injected", "signal": mock_signal}
