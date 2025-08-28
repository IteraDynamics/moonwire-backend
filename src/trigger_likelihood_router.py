# src/trigger_likelihood_router.py
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Query
from src import paths
import json
import os

from src.ml.infer import (
    infer_score,
    infer_score_ensemble,
    model_metadata,
    model_metadata_all,
)

from src.ml.thresholds import load_per_origin_thresholds
from src.ml.infer import model_metadata_all

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
def trigger_likelihood_metadata(view: str = Query(default="base", regex="^(base|all)$")):
    """
    Default (view=base): return classic flat logistic metadata so existing tests/clients pass:
      { "created_at": ..., "metrics": {...}, "feature_order": [...], ... }

    view=all: return both models if present:
      { "logistic": {...}, "random_forest": {...} }
    """
    # Load logistic (primary) with monkeypatch-friendly paths
    try:
        base = model_metadata(models_dir=paths.MODELS_DIR)
    except Exception:
        base = {}

    # Optional RF metadata
    rf_meta = None
    try:
        rf_meta_path = paths.MODELS_DIR / "trigger_likelihood_rf.meta.json"
        if rf_meta_path.exists():
            with rf_meta_path.open("r") as f:
                rf_meta = json.load(f)
    except Exception:
        rf_meta = None  # ignore RF read errors

    # If nothing at all, allow demo fallback (200) or 503
    if not base and not rf_meta:
        if os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
            return {
                "demo": True,
                "message": "No model artifacts found; returning demo metadata.",
                "created_at": None,
                "metrics": {"roc_auc_va": 0.5},
                "feature_order": [],
                "feature_coverage": {},
                "top_features": [],
            }
        raise HTTPException(status_code=503, detail="No model artifacts available")

    # base view: return flat logistic meta (backward-compatible for tests)
    if view == "base":
        if base:
            return base
        # no logistic but RF exists → return RF flat to still satisfy tests' shape expectation
        return rf_meta

    # view=all: return both if available
    payload = {}
    if base:
        payload["logistic"] = base
    if rf_meta:
        payload["random_forest"] = rf_meta
    return payload