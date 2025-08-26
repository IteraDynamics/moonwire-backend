# src/trigger_likelihood_router.py
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query
from typing import Any, Dict

from src.ml.infer import infer_score, infer_score_ensemble, model_metadata

router = APIRouter()  # main.py mounts with prefix="/internal"

def _fallback_contribs(payload: Dict[str, Any], top_n: int) -> Dict[str, float]:
    """
    If the scorer didn’t return contributions (e.g., older model or demo path),
    synthesize a tiny, deterministic contribution dict from the provided features.
    This is only a display/testing aid; real contributions still come from the model.
    """
    feats = payload.get("features") or {}
    if not isinstance(feats, dict) or not feats:
        return {"bias": 1.0}
    # take the largest |value| features as a stand-in
    items = sorted(
        ((k, float(feats.get(k, 0.0) or 0.0)) for k in feats.keys()),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:max(1, min(20, top_n))]
    return {k: v for k, v in items}

@router.post("/trigger-likelihood/score")
def trigger_likelihood_score(
    payload: Dict[str, Any] = Body(...),
    use: str = Query("logistic", pattern="^(logistic|ensemble)$"),
    explain: bool = Query(False),
    top_n: int = Query(5, ge=1, le=20),
):
    """
    POST /internal/trigger-likelihood/score?use=logistic|ensemble&explain=true&top_n=3
    Body: {"features": {...}}  (or {"origin": "...", "timestamp": "..."})
    """
    try:
        if use == "ensemble":
            try:
                res = infer_score_ensemble(payload, explain=explain, top_n=top_n)
            except TypeError:
                res = infer_score_ensemble(payload)
        else:
            try:
                res = infer_score(payload, explain=explain, top_n=top_n)
            except TypeError:
                res = infer_score(payload, explain=explain)

        # Ensure contributions exist if explain=True, even if scorer didn’t return them
        if explain and not isinstance(res.get("contributions"), dict):
            res = dict(res)
            res["contributions"] = _fallback_contribs(payload, top_n)
        return res
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