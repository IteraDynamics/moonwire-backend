# src/signal_router.py

from fastapi import APIRouter
from src.signal_composer import generate_signal
from src.signal_cache import save_latest_signal, get_latest_signal

router = APIRouter()

@router.get("/signals/mock")
def get_mock_signal():
    signal = generate_signal()
    save_latest_signal(signal)
    return signal

@router.get("/signals/latest")
def get_latest_cached_signal():
    latest = get_latest_signal()
    return latest or {"error": "No signal cached yet."}