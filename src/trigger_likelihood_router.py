# src/trigger_likelihood_router.py
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query
from typing import Any, Dict

from src.ml.infer import infer_score, infer_score_ensemble, model_metadata

router = APIRouter()  # main.py mounts with prefix="/internal"

@router.post("/trigger-likelihood/score")
def trigger_likelihood_score(
    payload: Dict[str, Any] = Body(...),
    use: str = Query("logistic", regex="^(logistic|ensemble)$"),
    explain: bool = Query(False),
    top_n: int = Query(5, ge=1, le=20),
):
    """
    POST /internal/trigger-likelihood/score?use=logistic|ensemble&explain=true&top_n=3
    Body: {"features": {...}}  (or {"origin": "...", "timestamp": "..."})
    """
    try:
        if use == "ensemble":
            # Back-compat in case infer_score_ensemble doesn't accept explain/top_n yet
            try:
                return infer_score_ensemble(payload, explain=explain, top_n=top_n)
            except TypeError:
                return infer_score_ensemble(payload)
        else:
            try:
                return infer_score(payload, explain=explain, top_n=top_n)
            except TypeError:
                return infer_score(payload, explain=explain)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"scoring error: {e}")

@router.get("/trigger-likelihood/metadata")
def trigger_likelihood_metadata():
    """
    GET /internal/trigger-likelihood/metadata
    Returns model meta (logistic, rf, gb if present), coverage, metrics, etc.
    """
    meta = model_metadata()
    if not meta:
        raise HTTPException(status_code=503, detail="model artifacts unavailable")
    return meta

__all__ = ["router"]