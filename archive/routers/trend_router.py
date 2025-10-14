# src/trend_router.py

from fastapi import APIRouter
from src.trend_deltas import load_signal_history, compute_trend_deltas

router = APIRouter(prefix="", tags=["trends"])

@router.get("/trend-deltas")
def get_trend_deltas():
    history = load_signal_history()
    deltas = compute_trend_deltas(history)
    return deltas
