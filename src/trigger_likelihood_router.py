# src/trigger_likelihood_router.py
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query
from typing import Any, Dict

# Logistic + ensemble inference helpers
from src.ml.infer import infer_score, infer_score_ensemble, model_metadata

router = APIRouter()  # no prefix here; main.py will mount with prefix="/internal"

@router.post("/trigger-likelihood/score")
def trigger_likelihood_score(
    payload: Dict[str, Any] = Body(...),
    use: str = Query("logistic", regex="^(logistic|ensemble)$"),
):
    """
    POST /internal/trigger-likelihood/score
    Body: {"features": {...}}  (or {"origin": "...", "timestamp": "..."})
    Query: use=logistic|ensemble
    """
    try:
        if use == "ensemble":
            out = infer_score_ensemble(payload)
        else:
            out = infer_score(payload)
        return out
    except Exception as e:
        # In test/demo, we prefer a 200 with a demo fallback only inside the model layer.
        # If we got here, something else is wrong.
        raise HTTPException(status_code=503, detail=f"scoring error: {e}")

@router.get("/trigger-likelihood/metadata")
def trigger_likelihood_metadata():
    """
    GET /internal/trigger-likelihood/metadata
    Returns model meta (logistic, rf, gb if present), coverage, metrics, etc.
    """
    meta = model_metadata()
    if not meta:
        # Tests expect 503 when artifacts are absent (unless demo path is used by score)
        raise HTTPException(status_code=503, detail="model artifacts unavailable")
    return meta

__all__ = ["router"]