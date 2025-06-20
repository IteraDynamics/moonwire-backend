# src/adjustment_trigger_router.py

from fastapi import APIRouter
from src.adjust_signals_based_on_feedback import adjust_signals

router = APIRouter()

@router.post("/internal/adjust-signals-based-on-feedback")
def trigger_adjust_signals():
    return adjust_signals()