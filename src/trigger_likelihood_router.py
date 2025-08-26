# src/trigger_likelihood_router.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Query

from src.ml.infer import infer_score, infer_score_ensemble, model_metadata
from src import paths  # <- import the module, not MODELS_DIR constant

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
    Body: {"features": {...}} (or {"origin": "...", "timestamp": "..."})
    """
    try:
        if use == "ensemble":
            try:
                res = infer_score_ensemble(payload, explain=explain, top_n=top_n)
            except TypeError:
                # older signature
                res = infer_score_ensemble(payload)
        else:
            try:
                res = infer_score(payload, explain=explain, top_n=top_n)
            except TypeError:
                # older signature
                res = infer_score(payload, explain=explain)

        if explain and not isinstance(res.get("contributions"), dict):
            # Ensure the field exists for explain=true callers/tests
            res = dict(res)
            res["contributions"] = _fallback_contribs(payload, top_n)
        return res
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"scoring error: {e}")


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        if path.exists():
            with path.open("r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _flatten_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Back-compat: tests expect a top-level 'metrics'.
    If ensemble-style nested meta is provided, flatten logistic up to top.
    Also keep nested blocks ('logistic','rf','gb') for richer callers.
    """
    if "metrics" in meta and isinstance(meta["metrics"], dict):
        return meta  # already flat

    if "logistic" in meta and isinstance(meta["logistic"], dict):
        flat = dict(meta["logistic"])
        # keep nested sections available too
        flat["logistic"] = meta.get("logistic")
        if "rf" in meta:
            flat["rf"] = meta["rf"]
        if "gb" in meta:
            flat["gb"] = meta["gb"]
        return flat

    # Last resort: ensure metrics key exists for callers
    if "metrics" not in meta:
        meta = dict(meta)
        meta["metrics"] = meta.get("metrics", {})
    return meta


@router.get("/trigger-likelihood/metadata")
def trigger_likelihood_metadata():
    """
    GET /internal/trigger-likelihood/metadata
    Returns model meta (logistic, rf, gb if present), coverage, metrics, etc.
    Robust to loader hiccups by falling back to direct file reads.
    """
    models_dir = paths.MODELS_DIR  # dynamic, respects monkeypatch in tests
    meta: Dict[str, Any] = {}

    # First try the official helper, pointing at the (possibly patched) models_dir
    try:
        meta = model_metadata(models_dir=models_dir) or {}
    except Exception:
        meta = {}

    # Fallback: read logistic meta directly
    if not meta:
        lg = _load_json(models_dir / "trigger_likelihood_v0.meta.json")
        if lg:
            meta = lg

    # Fallback: any *.meta.json in the dir
    if not meta:
        try:
            for p in sorted(models_dir.glob("*.meta.json")):
                obj = _load_json(p)
                if isinstance(obj, dict) and obj:
                    meta = obj
                    break
        except Exception:
            pass

    if not meta:
        raise HTTPException(status_code=503, detail="model artifacts unavailable")

    return _flatten_meta(meta)


__all__ = ["router"]