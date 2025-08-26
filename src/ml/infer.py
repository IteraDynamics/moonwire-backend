# src/ml/infer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np

from src.paths import (
    MODELS_DIR,
    RETRAINING_LOG_PATH,
    RETRAINING_TRIGGERED_LOG_PATH,
)
from src.analytics.origin_utils import normalize_origin as _norm


# ---- Filenames ----------------------------------------------------------------

# Logistic baseline artifacts
_LR_MODEL = "trigger_likelihood_v0.joblib"
_LR_META  = "trigger_likelihood_v0.meta.json"
# Shared coverage file (written once at train-time)
_COV_NAME = "feature_coverage.json"

# Random Forest artifacts
_RF_MODEL = "trigger_likelihood_rf.joblib"
_RF_META  = "trigger_likelihood_rf.meta.json"

# Gradient Boosting artifacts
_GB_MODEL = "trigger_likelihood_gb.joblib"
_GB_META  = "trigger_likelihood_gb.meta.json"


# ---- Small helpers ------------------------------------------------------------

def _artifact_paths(models_dir: Path | None, model_name: str) -> Tuple[Path, Path]:
    md = models_dir or MODELS_DIR
    if model_name == "logistic":
        return md / _LR_MODEL, md / _LR_META
    if model_name == "rf":
        return md / _RF_MODEL, md / _RF_META
    if model_name == "gb":
        return md / _GB_MODEL, md / _GB_META
    raise ValueError(f"unknown model_name={model_name}")

def _coverage_path(models_dir: Path | None) -> Path:
    return (models_dir or MODELS_DIR) / _COV_NAME

def _try_load(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def _load_model_and_meta(models_dir: Path | None, model_name: str):
    mpath, jpath = _artifact_paths(models_dir, model_name)
    model = _try_load(mpath)
    meta: Dict[str, Any] | None = None
    try:
        if jpath.exists():
            with jpath.open("r") as f:
                meta = json.load(f)
    except Exception:
        meta = None
    return model, meta

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

def _vectorize(features: Dict[str, Any], feat_order: List[str]) -> np.ndarray:
    return np.array(
        [[_safe_float(features.get(k, 0.0), 0.0) for k in feat_order]],
        dtype=float,
    )

def _contributions_linear(model, xrow: np.ndarray, feat_order: List[str], top_n: int | None) -> Dict[str, float]:
    """
    For linear models (logistic), return coef*value per feature.
    Trees/GB don't have simple linear contributions; we omit those here.
    """
    try:
        coef = model.coef_.ravel()
        contrib = {k: float(coef[i] * xrow[0, i]) for i, k in enumerate(feat_order)}
        if top_n is not None:
            items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
            return {k: float(v) for k, v in items}
        return contrib
    except Exception:
        return {}


# ---- Public metadata ----------------------------------------------------------

def model_metadata(models_dir: Path | None = None) -> Dict[str, Any]:
    """
    Return a merged metadata view.

    - Root contains logistic meta (for back-compat with older tests) if present:
      'metrics', 'feature_order', 'demo', 'artifacts', etc.
    - Also returns nested blocks for each available model:
      { 'logistic': {...}, 'rf': {...}, 'gb': {...} }
    - If a coverage file exists, expose it at root under 'feature_coverage'.
    """
    out: Dict[str, Any] = {}

    # Load metas
    _, lr_meta = _load_model_and_meta(models_dir, "logistic")
    _, rf_meta = _load_model_and_meta(models_dir, "rf")
    _, gb_meta = _load_model_and_meta(models_dir, "gb")

    # Root: prefer to seed with logistic for back-compat
    if lr_meta:
        out.update(lr_meta)

    # Attach coverage summary if present
    cov_path = _coverage_path(models_dir)
    if cov_path.exists():
        try:
            with cov_path.open("r") as f:
                out["feature_coverage"] = json.load(f)
        except Exception:
            pass

    # Nested blocks
    if lr_meta:
        out["logistic"] = lr_meta
        # Make sure top-level tests see `metrics` even when they expect legacy shape
        out.setdefault("metrics", lr_meta.get("metrics", {}))
    if rf_meta:
        out["rf"] = rf_meta
    if gb_meta:
        out["gb"] = gb_meta

    return out


# ---- Scoring (logistic) -------------------------------------------------------

def infer_score(
    payload: Dict[str, Any],
    *,
    explain: bool = False,
    top_n: int = 5,
    models_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Logistic-only scoring path (back-compat). Expects:
      payload = {"features": {...}}  (CI uses this form)
    """
    feats = payload.get("features")

    # Try to load logistic artifacts
    lr_model, lr_meta = _load_model_and_meta(models_dir, "logistic")

    # Demo fallback if no artifacts
    if lr_model is None or lr_meta is None:
        # Provide a deterministic-looking but obviously demo result
        base_p = _sigmoid(0.1 * _safe_float((feats or {}).get("burst_z", 0.0), 0.0)) if feats else 0.062
        out = {"prob_trigger_next_6h": float(base_p), "demo": True}
        if explain and feats:
            out["contributions"] = {"burst_z": float(0.1 * _safe_float(feats.get("burst_z", 0.0), 0.0))}
        return out

    # Normal path
    feat_order = lr_meta.get("feature_order") or []
    if not feats:
        # Keep the old safe value if someone posts origin-only
        return {"prob_trigger_next_6h": 0.062, "note": "no features in payload", "demo": bool(lr_meta.get("demo", False))}

    x = _vectorize(feats, feat_order)
    proba = float(lr_model.predict_proba(x)[0, 1])

    out = {"prob_trigger_next_6h": proba}
    if explain:
        out["contributions"] = _contributions_linear(lr_model, x, feat_order, top_n=top_n)
    return out

# Back-compat alias used by tests
def score(payload: dict, explain: bool = False):
    return infer_score(payload, explain=explain)


# ---- Ensemble scoring (logistic + rf + gb) -----------------------------------

def infer_score_ensemble(
    payload: Dict[str, Any],
    *,
    models_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "prob_trigger_next_6h": <mean>,
        "low": <band_low>,
        "high": <band_high>,
        "per_model": {"logistic": p1, "rf": p2, "gb": p3},
        "votes":     {"logistic": p1, "rf": p2, "gb": p3},  # alias for tests
        "demo": <bool>
      }
    """
    feats = payload.get("features") or {}

    # Load models/metas if present
    lr_model, lr_meta = _load_model_and_meta(models_dir, "logistic")
    rf_model, rf_meta = _load_model_and_meta(models_dir, "rf")
    gb_model, gb_meta = _load_model_and_meta(models_dir, "gb")

    votes: Dict[str, float] = {}
    feat_order: List[str] = []
    if lr_meta:
        feat_order = lr_meta.get("feature_order") or []

    # If no trained models present → demo fallback
    if (lr_model is None and rf_model is None and gb_model is None) or not feat_order:
        base_p = _sigmoid(0.1 * _safe_float(feats.get("burst_z", 0.0), 0.0)) if feats else 0.062
        votes = {"logistic": float(base_p)}
        probs = list(votes.values())
        return {
            "prob_trigger_next_6h": float(np.mean(probs)),
            "low": float(min(probs)),
            "high": float(max(probs)),
            "per_model": votes,
            "votes": votes,          # alias expected by tests
            "demo": True,
        }

    # Vectorize once
    x = _vectorize(feats, feat_order)

    # Logistic
    if lr_model is not None:
        try:
            votes["logistic"] = float(lr_model.predict_proba(x)[0, 1])
        except Exception:
            pass

    # Random Forest
    if rf_model is not None:
        try:
            votes["rf"] = float(rf_model.predict_proba(x)[0, 1])
        except Exception:
            pass

    # Gradient Boosting
    if gb_model is not None:
        try:
            votes["gb"] = float(gb_model.predict_proba(x)[0, 1])
        except Exception:
            pass

    # If somehow nothing scored, fallback to demo-ish logistic proxy
    if not votes:
        base_p = _sigmoid(0.1 * _safe_float(feats.get("burst_z", 0.0), 0.0)) if feats else 0.062
        votes = {"logistic": float(base_p)}
        probs = list(votes.values())
        return {
            "prob_trigger_next_6h": float(np.mean(probs)),
            "low": float(min(probs)),
            "high": float(max(probs)),
            "per_model": votes,
            "votes": votes,
            "demo": True,
        }

    probs = list(votes.values())
    mean_p = float(np.mean(probs))
    low_p  = float(min(probs))
    high_p = float(max(probs))

    return {
        "prob_trigger_next_6h": mean_p,
        "low": low_p,
        "high": high_p,
        "per_model": votes,
        "votes": votes,   # alias for tests / CI summary convenience
        "demo": False,
    }


# ---- Online inference / live backtest (last 24h) -----------------------------

def _load_jsonl(path: Path) -> List[dict]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except Exception:
        return []

def _parse_ts(v):
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        try:
            s = str(v)
            s = s[:-1] + "+00:00" if s.endswith("Z") else s
            return datetime.fromisoformat(s).astimezone(timezone.utc)
        except Exception:
            return None

def _label_has_trigger_between(triggers: List[dict], origin: str, t0: datetime, t1: datetime) -> int:
    o = _norm(origin)
    for r in triggers:
        if _norm(r.get("origin", "")) != o:
            continue
        ts = _parse_ts(r.get("timestamp"))
        if ts and t0 < ts <= t1:
            return 1
    return 0

def live_backtest_last_24h(interval: str = "hour", threshold: float = 0.5) -> Dict[str, Any]:
    """
    Score each origin at each hourly bucket over the last 24h and compute precision/recall snapshot.
    Uses the `score` (logistic) path; safe under demo.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [now - timedelta(hours=i) for i in range(24, 0, -1)]
    flags = _load_jsonl(RETRAINING_LOG_PATH)
    # Use origins seen in last 24h flags; fallback to a small default set
    recent_cut = now - timedelta(hours=24)
    origins = sorted({_norm(r.get("origin", "unknown"))
                      for r in flags
                      if (ts := _parse_ts(r.get("timestamp"))) and ts >= recent_cut}) or ["twitter", "reddit", "rss_news"]
    triggers = _load_jsonl(RETRAINING_TRIGGERED_LOG_PATH)

    per_origin = []
    for o in origins[:10]:
        tp = fp = fn = tn = 0
        for t in buckets:
            t_iso = t.isoformat()
            try:
                p = score({"origin": o, "timestamp": t_iso}).get("prob_trigger_next_6h", 0.0)
            except Exception:
                p = 0.0
            y = _label_has_trigger_between(triggers, o, t, t + timedelta(hours=6))
            yhat = 1 if p >= threshold else 0
            if   yhat == 1 and y == 1: tp += 1
            elif yhat == 1 and y == 0: fp += 1
            elif yhat == 0 and y == 1: fn += 1
            else: tn += 1
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        per_origin.append({
            "origin": o,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    # Overall
    tp = sum(po["tp"] for po in per_origin)
    fp = sum(po["fp"] for po in per_origin)
    fn = sum(po["fn"] for po in per_origin)
    prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / float(tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "window_hours": 24,
        "threshold": threshold,
        "overall": {"precision": round(prec, 3), "recall": round(rec, 3), "tp": tp, "fp": fp, "fn": fn},
        "per_origin": per_origin[:3],
    }


__all__ = [
    "infer_score",
    "score",
    "infer_score_ensemble",
    "model_metadata",
    "live_backtest_last_24h",
]