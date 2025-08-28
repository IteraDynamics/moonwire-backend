# src/ml/infer.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np

# Import the module, not constants, so monkeypatching works in tests.
from src import paths
from src.analytics.origin_utils import normalize_origin as _norm

# Artifact names
_LOGI_MODEL = "trigger_likelihood_v0.joblib"
_LOGI_META  = "trigger_likelihood_v0.meta.json"

_RF_MODEL   = "trigger_likelihood_rf.joblib"
_RF_META    = "trigger_likelihood_rf.meta.json"

_GB_MODEL   = "trigger_likelihood_gb.joblib"
_GB_META    = "trigger_likelihood_gb.meta.json"

_COV_NAME   = "feature_coverage.json"

# --------------------------------------------------------------------
# Cosmetic floor (OFF by default; enable with CI_SUMMARY_COSMETIC_FLOOR=1)
# --------------------------------------------------------------------
_COSMETIC_FLOOR = float(os.getenv("ENSEMBLE_COSMETIC_FLOOR", "0.021"))  # ~2.1%
_COSMETIC_CEIL  = 1.0 - _COSMETIC_FLOOR

def _floor_prob(p: float) -> float:
    try:
        p = float(p)
    except Exception:
        return _COSMETIC_FLOOR
    if p <= 0.0:
        return _COSMETIC_FLOOR
    if p >= 1.0:
        return _COSMETIC_CEIL
    return p

def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default).lower()
    return v in ("1", "true", "yes", "on")

# --------------------------
# Loaders / Metadata
# --------------------------
def _paths_for(kind: str, models_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    md = models_dir or paths.MODELS_DIR
    if kind == "logistic":
        return md / _LOGI_MODEL, md / _LOGI_META
    if kind == "rf":
        return md / _RF_MODEL, md / _RF_META
    if kind == "gb":
        return md / _GB_MODEL, md / _GB_META
    raise ValueError(f"unknown model kind: {kind}")

def _coverage_path(models_dir: Optional[Path] = None) -> Path:
    return (models_dir or paths.MODELS_DIR) / _COV_NAME

def _load_one(kind: str, models_dir: Optional[Path] = None):
    """Return (model, meta) or (None, None)."""
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
    {
      "logistic": {...}, "rf": {...}, "gb": {...},
      "feature_coverage": {...}, "artifacts": {"feature_coverage": "..."}
    }
    """
    out: Dict[str, Any] = {}
    for kind in ("logistic", "rf", "gb"):
        _, meta = _load_one(kind, models_dir)
        if meta:
            out[kind] = meta

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
# Vectorization / utils
# --------------------------
def _vectorize(features: Dict[str, Any], feat_order: List[str]) -> np.ndarray:
    row = []
    for k in feat_order:
        v = features.get(k, 0.0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        # sanitize: NaN / +/-Inf → 0
        if not np.isfinite(v):
            v = 0.0
        row.append(v)
    return np.array([row], dtype=float)

def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _contributions_logistic(model, xrow: np.ndarray, feat_order: List[str], top_n: Optional[int]) -> Dict[str, float]:
    try:
        coef = model.coef_.ravel()
    except Exception:
        return {}
    contrib = {feat_order[i]: float(coef[i] * xrow[0, i]) for i in range(len(feat_order))}
    if top_n is None:
        return contrib
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return {k: float(v) for k, v in items}

# --------------------------
# Core logistic inference
# --------------------------
def infer_score(
    payload: Dict[str, Any],
    *,
    explain: bool = False,
    top_n: int = 5,
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Logistic-only scoring path (back-compat)."""
    feats = payload.get("features") or {}

    model, meta = _load_one("logistic", models_dir)
    if model and meta:
        order = meta.get("feature_order") or []
        if order:
            x = _vectorize(feats, order)
            proba = float(model.predict_proba(x)[0, 1])
            out = {"prob_trigger_next_6h": proba}
            if explain:
                out["contributions"] = _contributions_logistic(model, x, order, top_n=top_n)
            return out

    # Demo fallback: simple burst_z proxy
    p = float(_sigmoid(0.1 * _safe_float(feats.get("burst_z", 0.0), 0.0)))
    out = {"prob_trigger_next_6h": p, "demo": True}
    if explain:
        out["contributions"] = {"burst_z": float(0.1 * _safe_float(feats.get("burst_z", 0.0), 0.0))}
    return out

def score(payload: Dict[str, Any], explain: bool = False):
    """Backward-compatible alias used by tests."""
    return infer_score(payload, explain=explain)

# --------------------------
# Ensemble inference
# --------------------------
def infer_score_ensemble(
    payload: Dict[str, Any],
    *,
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Average available learners (logistic, rf, gb).
    IMPORTANT: RF/GB are only included if the payload has enough non-zero features
    for that model's feature_order (prevents averaging in meaningless 0.0 votes).
    """
    feats = payload.get("features") or {}
    votes: Dict[str, float] = {}
    used: List[str] = []

    debug = os.getenv("MW_DEBUG_ENS", "0") in ("1","true","yes")
    errors = {}
    ...
    try:
        votes["rf"] = float(rf_model.predict_proba(x)[0,1]); used.append("rf")
    except Exception as e:
        if debug: errors["rf"] = f"{type(e).__name__}: {e}"
    ...
    try:
        votes["gb"] = float(gb_model.predict_proba(x)[0,1]); used.append("gb")
    except Exception as e:
        if debug: errors["gb"] = f"{type(e).__name__}: {e}"
    ...
    out = { ... }
    if debug and errors:
        out["debug_errors"] = errors
    return out



    

    def _nz_count(order: List[str]) -> int:
        # how many features are actually present / non-zero for this model?
        c = 0
        for k in order:
            try:
                v = float(feats.get(k, 0.0) or 0.0)
            except Exception:
                v = 0.0
            if abs(v) > 1e-12:
                c += 1
        return c

    # --- logistic (always allowed; falls back to demo if artifacts missing) ---
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

    # --- rf (require a few non-zero features to contribute) ---
    rf_model, rf_meta = _load_one("rf", models_dir)
    if rf_model and rf_meta:
        order = rf_meta.get("feature_order") or []
        if order and _nz_count(order) >= 3:
            x = _vectorize(feats, order)
            try:
                p_rf = float(rf_model.predict_proba(x)[0, 1])
                votes["rf"] = p_rf
                used.append("rf")
            except Exception:
                pass  # do not inject zeros

    # --- gb (same gating) ---
    gb_model, gb_meta = _load_one("gb", models_dir)
    if gb_model and gb_meta:
        order = gb_meta.get("feature_order") or []
        if order and _nz_count(order) >= 3:
            x = _vectorize(feats, order)
            try:
                p_gb = float(gb_model.predict_proba(x)[0, 1])
                votes["gb"] = p_gb
                used.append("gb")
            except Exception:
                pass

    # If nothing usable voted, provide a safe, clearly-marked demo fallback.
    if not votes:
        # demo-ish logistic proxy on burst_z only
        bz = 0.0
        try:
            bz = float(feats.get("burst_z", 0.0) or 0.0)
        except Exception:
            bz = 0.0
        p = float(_sigmoid(0.1 * bz))
        return {
            "prob_trigger_next_6h": p,
            "low": p,
            "high": p,
            "votes": {"logistic": p},
            "per_model": {"logistic": p},
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
        "votes": votes,         # preferred
        "per_model": votes,     # alias
        "models_used": used,
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
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        pass
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
    a precision/recall snapshot. Uses logistic `score` per bucket.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [now - timedelta(hours=i) for i in range(24, 0, -1)]

    flags = _load_jsonl(paths.RETRAINING_LOG_PATH)
    triggers = _load_jsonl(paths.RETRAINING_TRIGGERED_LOG_PATH)

    recent_cut = now - timedelta(hours=24)
    origins = sorted({
        _norm(r.get("origin", "unknown"))
        for r in flags
        if (ts := _parse_ts(r.get("timestamp"))) and ts >= recent_cut
    }) or ["twitter", "reddit", "rss_news"]

    per_origin: List[Dict[str, Any]] = []
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
