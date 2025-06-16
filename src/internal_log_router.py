# src/internal_log_router.py

from fastapi import APIRouter
from datetime import datetime  # ✅ Required for timestamp
from src.auto_log_signals import run_signal_logger

router = APIRouter(prefix="/internal", tags=["internal-tools"])

@router.post("/log-signals")
def trigger_signal_logging():
    run_signal_logger()
    return {"status": "logged", "source": "mock", "timestamp": str(datetime.utcnow())}