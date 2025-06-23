from fastapi import APIRouter
import sys
from pathlib import Path

# Add project root to sys.path so "scripts" folder is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.predict_disagreement import predict_disagreement  # ✅ Updated import

router = APIRouter()

@router.post("/internal/adjust-signals-based-on-feedback")
def trigger_adjust_signals():
    return predict_disagreement()