# src/trigger_likelihood_router.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Any, Dict

from src.ml.infer import infer_score, infer_score_ensemble, model_metadata_all

router = APIRouter(prefix="/internal/trigger-likelihood", tags=["trigger_likelihood"])


@router.post("/score")
def post_score(payload: Dict[str, Any], use: str = Query("logistic", enum=["logistic", "ensemble"])) -> Dict[str, Any]:
    try:
        if use == "ensemble":
            return infer_score_ensemble(payload)
        return infer_score(payload)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"inference failed: {e}")


@router.get("/metadata")
def get_metadata() -> Dict[str, Any]:
    meta_all = model_metadata_all()
    if not meta_all:
        raise HTTPException(status_code=503, detail="model artifacts unavailable")
    # back-compat for older tests: expose top-level "metrics" if logistic present
    return meta_all
