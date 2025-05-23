from fastapi import APIRouter
from src.signal_composer import generate_signal
from src.signal_log import log_composite_signal

router = APIRouter()

@router.get("/signals/mock")
def get_mock_signal():
    signal = generate_signal(
        asset="BTC",
        sentiment_score=0.74,
        price_at_score=68100,
        fallback_type="mock"
    )
    log_composite_signal(signal)
    return signal