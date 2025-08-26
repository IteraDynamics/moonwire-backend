# src/ml/infer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np

from src.paths import (
    MODELS_DIR,
    RETRAINING_LOG_PATH,
    RETRAINING_TRIGGERED_LOG_PATH,
)
from src.analytics.origin_utils import normalize_origin as _norm


# --------------------------
# Artifact names (by model)
# --------------------------
_LOGI_MODEL = "trigger_likelihood_v0.joblib"
_LOGI_META  = "trigger_likelihood_v0.meta.json"

_RF_MODEL   = "trigger_likelihood_rf.joblib"
_RF_META    = "trigger_likelihood_rf.meta.json"

_GB_MODEL   = "trigger_likelihood_gb.joblib"
_GB_META    = "trigger_likelihood_gb.meta.json"

_COV_NAME   = "feature_coverage.json"


# --------------------------
# Loaders / Metadata
# --------------------------
def _paths_for(kind: str, models_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    md = models_dir or MODELS_DIR
    if kind == "logistic":
        return md / _LOGI_MODEL, md / _LOGI_META
    if kind == "rf":
        return md / _RF_MODEL, md / _RF_META
    if kind == "gb":
        return md / _GB_MODEL, md / _GB_META
    raise ValueError(f"unknown model kind: {kind}")


def _coverage_path(models_dir: Optional[Path] = None) -> Path:
    return (models_dir or MODELS_DIR) / _COV_NAME


def _load_one(kind: str, models_dir: Optional[Path] = None):
    """Try to load a single model + its meta; return (model, meta) or (None, None)."""
    mpath, jpath = _paths_for(kind, models_dir)
    try:
        model = joblib.load(mpath)
    except Exception:
        return None, None
    try:
        with jpath.open("r") as f:
            meta = json.load(f)
    except Exception:
        meta = {}
    return model, meta


def model_metadata(models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Return a nested metadata blob:
      {
        "logistic": {...}, "rf": {...}, "gb": {...},
        "feature_coverage": {...}, "artifacts": {"feature_coverage": "..."}
      }
    Missing learners are simply omitted.
    """
    out: Dict[str, Any] = {}

    # Learners
    for kind in ("logistic", "rf", "gb"):
        _, meta = _load_one(kind, models_dir)
        if meta:
            out[kind] = meta

    # Coverage (shared)
    cov_path = _coverage_path(models_dir)
    coverage = {}
    if cov_path.exists():
        try:
            with cov_path.open("r") as f:
                coverage = json.load(f)
        except Exception:
            coverage = {}
    if coverage:
        out["feature_coverage"] = coverage
    out["artifacts"] = {"feature_coverage": str(cov_path)}

    return out


# --------------------------
# Vectorization / math utils
# --------------------------
def _vectorize(features: Dict[str, Any], feat_order: List[str]) -> np.ndarray:
    return np.array(
        [[float(features.get(k, 0.0) or 0.0) for k in feat_order]],
        dtype=float,
    )


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _contributions_logistic(model, xrow: np.ndarray, feat_order: List[str], top_n: Optional[int]) -> Dict[str, float]:
    """coef * value for logistic models; sorted by |contrib| if top_n is set."""
    try:
        coef = model.coef_.ravel()
    except Exception:
        return {}
    contrib = {feat_order[i]: float(coef[i] * xrow[0, i]) for i in range(len(feat_order))}
    if top_n is None:
        return contrib
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return {k: float(v) for k, v in items}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


# --------------------------
# Core inference (logistic)
# --------------------------
def infer_score(
    payload: Dict[str, Any],
    *,
    explain: bool = False,
    top_n: int = 5,
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Logistic-only scoring path (backward compatible).
    Accepts payloads of the form {"features": {...}}. (Origin path not wired here.)
    """
    feats = payload.get("features") or {}

    # Try real artifacts first
    model, meta = _load_one("logistic", models_dir)
    if model and meta:
        feat_order = meta.get("feature_order") or []
        if feat_order:
            x = _vectorize(feats, feat_order)
            proba = float(model.predict_proba(x)[0, 1])
            out = {"prob_trigger_next_6h": proba}
            if explain:
                out["contributions"] = _contributions_logistic(model, x, feat_order, top_n=top_n)
            return out

    # Demo fallback if model missing
    p = float(_sigmoid(0.1 * _safe_float(feats.get("burst_z", 0.0), 0.0)))
    out = {"prob_trigger_next_6h": p, "demo": True}
    if explain:
        out["contributions"] = {"burst_z": float(0.1 * _safe_float(feats.get("burst_z", 0.0), 0.0))}
    return out


# Back-compat name used by tests
def score(payload: Dict[str, Any], explain: bool = False):
    return infer_score(payload, explain=explain)


# --------------------------
# Ensemble inference (logistic + rf + gb)
# --------------------------
def infer_score_ensemble(
    payload: Dict[str, Any],
    *,
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Average the probabilities from all available learners.
    Returns:
      {
        "prob_trigger_next_6h": mean,
        "low": min_prob,
        "high": max_prob,
        "votes": {"logistic": p, "rf": p?, "gb": p?},
        "per_model": {...},          # alias for convenience / back-compat
        "models_used": ["logistic", ...],
        "demo": False|True
      }
    """
    feats = payload.get("features") or {}
    votes: Dict[str, float] = {}
    used: List[str] = []

    # logistic
    l_model, l_meta = _load_one("logistic", models_dir)
    if l_model and l_meta:
        order = l_meta.get("feature_order") or []
        if order:
            x = _vectorize(feats, order)
            try:
                votes["logistic"] = float(l_model.predict_proba(x)[0, 1])
                used.append("logistic")
            except Exception:
                pass

    # rf
    rf_model, rf_meta = _load_one("rf", models_dir)
    if rf_model and rf_meta:
        order = rf_meta.get("feature_order") or []
        if order:
            x = _vectorize(feats, order)
            try:
                # RF supports predict_proba
                votes["rf"] = float(rf_model.predict_proba(x)[0, 1])
                used.append("rf")
            except Exception:
                pass

    # gb
    gb_model, gb_meta = _load_one("gb", models_dir)
    if gb_model and gb_meta:
        order = gb_meta.get("feature_order") or []
        if order:
            x = _vectorize(feats, order)
            try:
                # GradientBoostingClassifier supports predict_proba
                votes["gb"] = float(gb_model.predict_proba(x)[0, 1])
                used.append("gb")
            except Exception:
                pass

    # No learners available -> demo-safe proxy based on burst_z
    if not votes:
        p = float(_sigmoid(0.1 * _safe_float(feats.get("burst_z", 0.0), 0.0)))
        return {
            "prob_trigger_next_6h": p,
            "low": p,
            "high": p,
            "votes": {"demo": p},
            "per_model": {"demo": p},
            "models_used": [],
            "demo": True,
        }

    probs = list(votes.values())
    mean_p = float(sum(probs) / len(probs))
    low_p  = float(min(probs))
    high_p = float(max(probs))

    return {
        "prob_trigger_next_6h": mean_p,
        "low": low_p,
        "high": high_p,
        "votes": votes,          # preferred key
        "per_model": votes,      # alias for router/CI summary convenience
        "models_used": used,     # e.g., ["logistic", "rf", "gb"]
        "demo": False,
    }


# --------------------------
# Live backtest (last 24h)
# --------------------------
def _load_jsonl(path: Path) -> List[dict]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except Exception:
        return []


def _parse_ts(v: Any) -> Optional[datetime]:
    # epoch seconds
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        pass
    # ISO (with optional Z)
    try:
        s = str(v)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
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
    Score each origin at each hourly bucket over the last 24h and compute
    a precision/recall snapshot. Uses `score` (logistic) per historical bucket.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [now - timedelta(hours=i) for i in range(24, 0, -1)]

    flags = _load_jsonl(RETRAINING_LOG_PATH)
    triggers = _load_jsonl(RETRAINING_TRIGGERED_LOG_PATH)

    # Prefer recent origins present in flags; fallback to the standard three
    recent_cut = now - timedelta(hours=24)
    origins = sorted({
        _norm(r.get("origin", "unknown"))
        for r in flags
        if (ts := _parse_ts(r.get("timestamp"))) and ts >= recent_cut
    }) or ["twitter", "reddit", "rss_news"]

    per_origin: List[Dict[str, Any]] = []
    for o in origins[:10]:  # cap
        tp = fp = fn = tn = 0
        for t in buckets:
            t_iso = t.isoformat()
            try:
                p = score({"origin": o, "timestamp": t_iso}).get("prob_trigger_next_6h", 0.0)
            except Exception:
                p = 0.0
            y = _label_has_trigger_between(triggers, o, t, t + timedelta(hours=6))
            yhat = 1 if p >= threshold else 0
            if yhat == 1 and y == 1:
                tp += 1
            elif yhat == 1 and y == 0:
                fp += 1
            elif yhat == 0 and y == 1:
                fn += 1
            else:
                tn += 1
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        per_origin.append({
            "origin": o,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

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
    "infer_score_ensemble",
    "score",
    "model_metadata",
    "live_backtest_last_24h",
]