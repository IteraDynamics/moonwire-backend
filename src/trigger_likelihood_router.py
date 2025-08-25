# src/trigger_likelihood_router.py
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Query

from src.ml.infer import (
    infer_score,
    infer_score_ensemble,
    model_metadata,
    model_metadata_all,
)

# This router is expected to be mounted in main.py under prefix="/internal"
router = APIRouter(tags=["trigger_likelihood"])


@router.post("/trigger-likelihood/score")
def score_endpoint(
    payload: Dict[str, Any] = Body(...),
    use: str = Query("logistic", regex="^(logistic|ensemble)$"),
    explain: bool = Query(False),
):
    """
    POST body can be either:
      { "features": {...} }                          # preferred
      { "origin": "twitter", "timestamp": "..." }    # best-effort fallback
    Query params:
      use=logistic|ensemble   -> choose scorer
      explain=true            -> include linear contributions (logistic only)
    """
    try:
        if use == "ensemble":
            return infer_score_ensemble(payload)
        # default to logistic
        return infer_score(payload, explain=explain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scoring error: {e}")


@router.get("/trigger-likelihood/metadata")
def metadata_endpoint():
    """
    Returns metadata for available models.
    Shape:
      { "models": { "logistic": {...}, "rf": {...} } }
    Falls back to logistic-only; 503 if nothing present.
    """
    all_meta = model_metadata_all()
    if all_meta:
        return {"models": all_meta}

    L = model_metadata()
    if L:
        return {"models": {"logistic": L}}

    raise HTTPException(status_code=503, detail="Model metadata unavailable")