# src/ml/infer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

from src.paths import MODELS_DIR

_MODEL_NAME = "trigger_likelihood_v0.joblib"
_META_NAME = "trigger_likelihood_v0.meta.json"
_COV_NAME = "feature_coverage.json"


def _artifact_paths(models_dir: Path | None = None) -> Tuple[Path, Path, Path]:
    md = models_dir or MODELS_DIR
    return md / _MODEL_NAME, md / _META_NAME, md / _COV_NAME


def _load_model_and_meta(models_dir: Path | None = None):
    mpath, jpath, cpath = _artifact_paths(models_dir)
    model = joblib.load(mpath)
    with jpath.open("r") as f:
        meta = json.load(f)
    cov = {}
    if cpath.exists():
        try:
            with cpath.open("r") as f:
                cov = json.load(f)
        except Exception:
            cov = {}
    return model, meta, cov


def model_metadata(models_dir: Path | None = None) -> Dict[str, Any]:
    """Return merged metadata: meta.json + (optional) feature_coverage.json."""
    try:
        _, meta, cov = _load_model_and_meta(models_dir)
        out = dict(meta)
        if cov:
            out["feature_coverage"] = cov
        return out
    except Exception:
        return {}


def _vectorize(features: Dict[str, Any], feat_order: List[str]) -> np.ndarray:
    return np.array([[float(features.get(k, 0.0) or 0.0) for k in feat_order]], dtype=float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _contributions(model, xrow: np.ndarray, feat_order: List[str], top_n: int | None) -> Dict[str, float]:
    """Return contributions (coef * value) keyed by feature; sorted if top_n is provided."""
    contrib: Dict[str, float] = {}
    try:
        coef = model.coef_.ravel()
        for i, k in enumerate(feat_order):
            contrib[k] = float(coef[i] * xrow[0, i])
        if top_n is not None:
            items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
            return {k: float(v) for k, v in items}
        return contrib
    except Exception:
        return {}


def infer_score(payload: Dict[str, Any], *, explain: bool = False, top_n: int = 5, models_dir: Path | None = None) -> Dict[str, Any]:
    """
    Score either:
      A) {"origin": "...", "timestamp": "..."}  -> (relies on upstream feature builder in your stack)
      B) {"features": {...}}                    -> direct features dict
    For CI summary we mostly use B).
    """
    try:
        model, meta, _ = _load_model_and_meta(models_dir)
    except Exception:
        # Demo fallback when no artifacts present
        if payload.get("features"):
            feats = payload["features"]
            p = 1 / (1 + np.exp(-0.1 * float(feats.get("burst_z", 0.0))))
            demo_res = {"prob_trigger_next_6h": float(p), "demo": True}
            if explain:
                demo_res["contributions"] = {"burst_z": float(0.1 * feats.get("burst_z", 0.0))}
            return demo_res
        return {"prob_trigger_next_6h": 0.062, "demo": True}

    feat_order = meta.get("feature_order") or []
    # Option A is not built out here—your summary already passes features.
    feats = payload.get("features")
    if feats is None:
        # Maintain compatibility: if only origin/timestamp supplied, return a safe value
        return {"prob_trigger_next_6h": 0.062, "note": "origin path not wired in infer", "demo": meta.get("demo", False)}

    x = _vectorize(feats, feat_order)
    proba = float(model.predict_proba(x)[0, 1])

    out = {"prob_trigger_next_6h": proba}
    if explain:
        out["contributions"] = _contributions(model, x, feat_order, top_n=top_n)
    return out

# --- Back-compat shim for tests & callers expecting `score` ---
def score(payload: dict, explain: bool = False):
    """
    Backward-compatible alias for infer_score.
    Tests import `from src.ml.infer import score`, so keep this symbol.
    """
    return infer_score(payload, explain=explain)

# Optional: make exports explicit
__all__ = [
    "infer_score",
    "score",
    "model_metadata",  # if you expose this helper
]

