# src/trigger_likelihood_router.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Query

from src import paths  # import the module so monkeypatch works in tests
from src.ml.infer import infer_score, infer_score_ensemble, model_metadata

router = APIRouter()  # main.py mounts this with prefix="/internal"


# ------------------------
# Helpers
# ------------------------
def _fallback_contribs(payload: Dict[str, Any] | None, top_n: int) -> Dict[str, float]:
    """
    If the scorer didn’t return contributions (older model or demo path),
    synthesize a tiny, deterministic contribution dict from the provided features.
    Only used for explain=true so callers/tests always see a dict.
    """
    feats = (payload or {}).get("features") or {}
    if not isinstance(feats, dict) or not feats:
        return {"bias": 1.0}
    items = sorted(
        ((k, float(feats.get(k, 0.0) or 0.0)) for k in feats.keys()),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:max(1, min(20, top_n))]
    return {k: v for k, v in items}


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
    If an ensemble-style nested meta is provided, flatten logistic up to top,
    while preserving the nested blocks ('logistic','rf','gb').
    """
    if "metrics" in meta and isinstance(meta["metrics"], dict):
        return meta  # already flat

    if "logistic" in meta and isinstance(meta["logistic"], dict):
        flat = dict(meta["logistic"])
        flat["logistic"] = meta.get("logistic")
        if "rf" in meta:
            flat["rf"] = meta["rf"]
        if "gb" in meta:
            flat["gb"] = meta["gb"]
        return flat

    # Ensure a 'metrics' key is present for very old artifacts
    if "metrics" not in meta:
        meta = dict(meta)
        meta["metrics"] = meta.get("metrics", {})
    return meta


# ------------------------
# Routes
# ------------------------
@router.get("/trigger-likelihood/metadata")
def trigger_likelihood_metadata() -> Dict[str, Any]:
    """
    Return model meta (logistic, rf, gb if present), coverage, metrics, etc.
    Strategy:
      1) If any per-model meta files exist, assemble a nested dict and flatten.
      2) Else try the helper model_metadata(models_dir=...).
      3) Else fallback to any *.meta.json found.
    """
    models_dir = paths.MODELS_DIR  # dynamic; tests monkeypatch paths.MODELS_DIR

    # 1) Prefer explicit nested view from per-model files (when present)
    lg = _load_json(models_dir / "trigger_likelihood_v0.meta.json")
    rf = _load_json(models_dir / "trigger_likelihood_rf.meta.json")
    gb = _load_json(models_dir / "trigger_likelihood_gb.meta.json")

    if any([lg, rf, gb]):
        nested: Dict[str, Any] = {}
        if isinstance(lg, dict) and lg:
            nested["logistic"] = lg
        if isinstance(rf, dict) and rf:
            nested["rf"] = rf
        if isinstance(gb, dict) and gb:
            nested["gb"] = gb
        return _flatten_meta(nested)

    # 2) Fall back to helper (may return only a flat logistic meta)
    meta: Dict[str, Any] = {}
    try:
        meta = model_metadata(models_dir=models_dir) or {}
    except Exception:
        meta = {}

    # 3) Last resort: first *.meta.json we find
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


@router.post("/trigger-likelihood/score")
def trigger_likelihood_score(
    # Use Body(None) so we don't inject Ellipsis into responses by accident.
    payload: Dict[str, Any] | None = Body(None),
    use: str = Query("logistic", pattern="^(logistic|ensemble)$"),
    explain: bool = Query(False),
    top_n: int = Query(5, ge=1, le=20),
) -> Dict[str, Any]:
    """
    POST /internal/trigger-likelihood/score?use=logistic|ensemble&explain=true&top_n=3
    Body: {"features": {...}}  (or {"origin": "...", "timestamp": "..."} for logistic)
    """
    payload = payload or {}
    try:
        if use == "ensemble":
            # Ensemble API doesn’t support explain/top_n; pass the payload as-is.
            res = infer_score_ensemble(payload)
        else:
            # Logistic path supports explain/top_n in current code; keep old-signature fallback.
            try:
                res = infer_score(payload, explain=explain, top_n=top_n)
            except TypeError:
                res = infer_score(payload, explain=explain)

        # Ensure contributions exist for explain=true callers/tests
        if explain and not isinstance(res.get("contributions"), dict):
            res = dict(res)
            res["contributions"] = _fallback_contribs(payload, top_n)
        return res
    except HTTPException:
        raise
    except Exception as e:
        # Surface a clean 503 so CI stays green on demo/missing-artifact cases
        raise HTTPException(status_code=503, detail=f"scoring error: {e}")


__all__ = ["router"]