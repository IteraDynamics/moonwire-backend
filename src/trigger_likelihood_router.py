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
def trigger_likelihood_metadata():
    """
    Return metadata for available trigger-likelihood models.
    - Always try logistic metadata (primary).
    - Optionally attach random-forest metadata if present.
    - Never 503 just because RF/ensemble is missing.
    - Respect tests that monkeypatch paths.MODELS_DIR by passing it through.
    """
    payload = {}

    # 1) Logistic (primary)
    try:
        base = model_metadata(models_dir=paths.MODELS_DIR)
    except Exception:
        base = {}

    if base:
        payload["logistic"] = base

    # 2) Random Forest (optional)
    try:
        rf_meta_path = paths.MODELS_DIR / "trigger_likelihood_rf.meta.json"
        if rf_meta_path.exists():
            with rf_meta_path.open("r") as f:
                payload["random_forest"] = json.load(f)
    except Exception:
        # Don't fail the request just because RF isn't readable
        pass

    # 3) If nothing at all, return demo metadata (200) in DEMO_MODE; otherwise 503
    if not payload:
        if os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
            return {
                "demo": True,
                "message": "No model artifacts found; returning demo metadata.",
                "logistic": {
                    "created_at": None,
                    "metrics": {"roc_auc_va": 0.5},
                    "feature_order": [],
                    "feature_coverage": {},
                    "top_features": [],
                },
            }
        # No artifacts and not in demo mode → service unavailable
        raise HTTPException(status_code=503, detail="No model artifacts available")

    return payload