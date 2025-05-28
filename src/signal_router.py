# signal_router.py

from fastapi import APIRouter
from src.signal_composer import generate_signal
from src.signal_log import log_signal
from pathlib import Path
import json

router = APIRouter()

@router.get("/signals/mock")
def get_mock_signal():
    signal = generate_signal(
        asset="BTC",
        sentiment_score=0.84,
        fallback_type="mock",
        top_drivers=["social sentiment", "volume shift"]
    )
    log_signal(signal_data=signal)
    return signal

@router.get("/signals/log")
def get_signal_log():
    log_path = Path("logs/signal_history.jsonl")
    if not log_path.exists():
        return {"signals": []}

    signals = []
    with open(log_path, "r") as f:
        for line in f:
            try:
                signals.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    return {"signals": signals[-10:]}  # return last 10
