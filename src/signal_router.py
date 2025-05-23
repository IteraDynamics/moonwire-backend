from fastapi import APIRouter
from src.signal_composer import generate_signal

router = APIRouter()

@router.get("/signals/mock")
def get_mock_signal():
    return generate_signal(
        asset="BTC",
        sentiment_score=0.84,
        fallback_type="mock",
        top_drivers=["social sentiment", "volume shift"]
    )