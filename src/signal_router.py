from fastapi import APIRouter
from src.signal_composer import generate_signal
from src.signal_log import log_signal

router = APIRouter()

@router.get("/signals/mock")
def get_mock_signal():
    # Hardcoded test values for now
    signal = generate_signal(
        asset="BTC",
        sentiment_score=0.74,
        price_at_score=68100,
        fallback_type="mock"
    )

    # Log the full structured signal
    log_signal(**signal)

    return signal