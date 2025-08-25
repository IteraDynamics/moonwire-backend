# src/trigger_likelihood_router.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from src.ml.infer import (
    infer_score,                 # logistic path (v0 / v0.1 / v0.2)
    infer_score_ensemble,        # ensemble path (v0.3)
    model_metadata,              # logistic metadata (includes coverage, metrics, etc.)
)
from src.paths import MODELS_DIR

router = APIRouter()


def _normalize_use(value: str) -> str:
    v = (value or "logistic").strip().lower()
    if v not in ("logistic", "ensemble"):
        raise HTTPException(status_code=422, detail="Invalid 'use' parameter; must be 'logistic' or 'ensemble'.")
    return v


@router.post("/trigger-likelihood/score")
def score_trigger(
    body: dict,
    use: str = Query("logistic", description="Which model to use: 'logistic' or 'ensemble'"),
    explain: Optional[bool] = Query(False, description="If true (logistic only), include feature contributions."),
) -> Any:
    """
    Score trigger likelihood.
    - When use=logistic: returns {"prob_trigger_next_6h", ...} (and optional "contributions" if explain=true)
    - When use=ensemble: returns {"prob_trigger_next_6h", "low", "high", "votes", "demo"}
    """
    mode = _normalize_use(use)
    try:
        if mode == "ensemble":
            return infer_score_ensemble(body)
        # default: logistic
        return infer_score(body, explain=bool(explain))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"score failed: {e}")


@router.get("/trigger-likelihood/metadata")
def trigger_likelihood_metadata(
    use: str = Query("logistic", description="Which model to use: 'logistic' or 'ensemble'"),
) -> Any:
    """
    Return model metadata.
    - logistic: returns merged meta (metrics, feature_order, coverage, top_features, etc.)
    - ensemble: returns the raw ensemble meta (weights, per-model metrics, bootstrap bands, feature_order, etc.)
    """
    mode = _normalize_use(use)
    if mode == "logistic":
        meta = model_metadata()
        if not meta:
            # For metadata we intentionally 503 when artifacts are missing (tests rely on this behavior).
            raise HTTPException(status_code=503, detail="logistic metadata not available")
        return meta

    # Ensemble meta lives in MODELS_DIR / trigger_ensemble.meta.json
    meta_path: Path = MODELS_DIR / "trigger_ensemble.meta.json"
    if not meta_path.exists():
        # Keep parity with logistic behavior: surface as 503 if missing.
        raise HTTPException(status_code=503, detail="ensemble meta not found")
    try:
        return json.loads(meta_path.read_text())
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"failed to read ensemble meta: {e}")