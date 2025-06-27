# src/internal_trusted_signals_router.py

from fastapi import APIRouter
from src.signal_cache import get_latest_signal, save_latest_signal
from src.feedback_utils import get_feedback_summary_for_signal
from src.signal_utils import compute_trust_scores
from src.disagreement_utils import get_disagreement_probability  # ✅ FIXED IMPORT

router = APIRouter(prefix="/internal")

@router.get("/signals-with-trust")
def get_signals_with_trust():
    """
    Returns the latest signal enriched with trust metrics.
    If no signal is cached yet, returns a 404-style error message.
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
            "historical_agreement_rate": feedback.get("user_agrees_rate", 0.5),
            "predicted_disagreement_prob": disagreement_prob
        }
    }

    enriched = compute_trust_scores(signal, trust_data)
    return enriched


@router.post("/inject-mock-signal")
def inject_mock_signal():
    """
    TEMPORARY:
    Injects a mock signal into the in-memory cache
    """
    mock_signal = {
        "id": "mock-signal-001",
        "label": "Positive",
        "score": 0.85,
        "confidence": 0.6,
        "timestamp": "2025-06-26T09:30:00Z"
    }

    save_latest_signal(mock_signal)
    return {"status": "Mock signal injected"}