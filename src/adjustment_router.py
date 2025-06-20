# src/adjustment_router.py

from fastapi import APIRouter
from src.adjust_signals_based_on_feedback import adjust_signals

router = APIRouter()

@router.post("/internal/adjust-signals-based-on-feedback")
def trigger_signal_adjustment():
    return adjust_signals()