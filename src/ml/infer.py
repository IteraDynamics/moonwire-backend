# src/ml/infer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
from datetime import datetime, timedelta, timezone

from src.paths import (
    MODELS_DIR,
    RETRAINING_LOG_PATH,
    RETRAINING_TRIGGERED_LOG_PATH,
)
from src.analytics.origin_utils import normalize_origin as _norm


# ---- Artifact names ---------------------------------------------------------
# Logistic (baseline)
_LOGI_MODEL = "trigger_likelihood_v0.joblib"
_LOGI_META  = "trigger_likelihood_v0.meta.json"
_COV_NAME   = "feature_coverage.json"

# Random Forest
_RF_MODEL = "trigger_likelihood_rf.joblib"
_RF_META  = "trigger_likelihood_rf.meta.json"

# Gradient Boosting
_GB_MODEL = "trigger_likelihood_gb.joblib"
_GB_META  = "trigger_likelihood_gb.meta.json"


# ---- Low-level helpers ------------------------------------------------------
def _safe_joblib_load(p: Path):
    try:
        if p.exists():
            return joblib.load(p)
    except Exception:
        pass
    return None


def _artifact_paths(
    model_name: str, meta_name: str, models_dir: Optional[Path] = None
) -> Tuple[Path, Path]:
    md = models_dir or MODELS_DIR
    return md / model_name, md / meta_name


def _load_logistic(models_dir: Optional[Path] = None):
    """Return (model, meta, coverage_dict) or (None, {}, {})."""
    mpath, jpath = _artifact_paths(_LOGI_MODEL, _LOGI_META, models_dir)
    cpath = (models_dir or MODELS_DIR) / _COV_NAME

    model = _safe_joblib_load(mpath)
    meta: Dict[str, Any] = {}
    cov: Dict[str, Any] = {}

    try:
        if jpath.exists():
            with jpath.open("r") as f:
                meta = json.load(f)
    except Exception:
        meta = {}

    try:
        if cpath.exists():
            with cpath.open("r") as f:
                cov = json.load(f)
    except Exception:
        cov = {}

    return model, meta, cov


def _load_rf(models_dir: Optional[Path] = None):
    mpath, jpath = _artifact_paths(_RF_MODEL, _RF_META, models_dir)
    return _safe_joblib_load(mpath), _json_or_empty(jpath)


def _load_gb(models_dir: Optional[Path] = None):
    mpath, jpath = _artifact_paths(_GB_MODEL, _GB_META, models_dir)
    return _safe_joblib_load(mpath), _json_or_empty(jpath)


def _json_or_empty(p: Path) -> Dict[str, Any]:
    try:
        if p.exists():
            with p.open("r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _vectorize(features: Dict[str, Any], feat_order: List[str]) -> np.ndarray:
    return np.array([[float(features.get(k, 0.0) or 0.0) for k in feat_order]], dtype=float)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _contributions_linear(model, xrow: np.ndarray, feat_order: List[str], top_n: Optional[int]) -> Dict[str, float]:
    """
    Contributions for linear models: coef * value.
    For non-linear models (RF/GB) we skip contributions here (use 'n/a').
    """
    out: Dict[str, float] = {}
    try:
        coef = model.coef_.ravel()
        for i, k in enumerate(feat_order):
            out[k] = float(coef[i] * xrow[0, i])
        if top_n is not None:
            items = sorted(out.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
            return {k: float(v) for k, v in items}
        return out
    except Exception:
        return {}


# ---- Public metadata helper -------------------------------------------------
def model_metadata(models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Return a merged metadata view. Top-level mirrors the logistic meta
    for back-compat (tests expect 'metrics' at the root), while nested
    blocks expose per-model details when available.
    """
    logi_model, logi_meta, coverage = _load_logistic(models_dir)
    rf_model, rf_meta = _load_rf(models_dir)
    gb_model, gb_meta = _load_gb(models_dir)

    if not logi_meta:
        # No baseline artifacts -> let the router return 503
        return {}

    # Compose a back-compat top-level view from logistic meta
    out: Dict[str, Any] = {
        "metrics": logi_meta.get("metrics", {}),
        "feature_order": logi_meta.get("feature_order", []),
        "demo": bool(logi_meta.get("demo", False)),
        "artifacts": logi_meta.get("artifacts", {}),
        "top_features": logi_meta.get("top_features", []),
        "feature_coverage_summary": logi_meta.get("feature_coverage_summary", {}),
    }
    if coverage:
        out["feature_coverage"] = coverage

    # Nested blocks
    out["logistic"] = logi_meta
    if rf_meta:
        out["rf"] = rf_meta
    if gb_meta:
        out["gb"] = gb_meta
    return out


# ---- Baseline scoring (logistic) --------------------------------------------
def infer_score(
    payload: Dict[str, Any],
    *,
    explain: bool = False,
    top_n: int = 5,
    models_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Score with the logistic model by default. Supports:
      - {"features": {...}}  (preferred path)
      - {"origin": "...", "timestamp": "..."}  (not wired here; returns safe default)
    """
    try:
        model, meta, _ = _load_logistic(models_dir)
        if model is None or not meta:
            raise RuntimeError("missing logistic artifacts")
    except Exception:
        # Demo fallback
        feats = payload.get("features", {}) or {}
        p = _sigmoid(0.1 * float(feats.get("burst_z", 0.0)))
        out = {"prob_trigger_next_6h": float(p), "demo": True}
        if explain and feats:
            out["contributions"] = {"burst_z": float(0.1 * feats.get("burst_z", 0.0))}
        return out

    feat_order = meta.get("feature_order") or []
    feats = payload.get("features")
    if feats is None:
        # Keep old behavior if someone only passes origin/timestamp
        return {
            "prob_trigger_next_6h": 0.062,
            "note": "origin path not wired in infer",
            "demo": bool(meta.get("demo", False)),
        }

    x = _vectorize(feats, feat_order)
    proba = float(model.predict_proba(x)[0, 1])  # type: ignore[attr-defined]

    out: Dict[str, Any] = {"prob_trigger_next_6h": proba}
    if explain:
        out["contributions"] = _contributions_linear(model, x, feat_order, top_n)
    return out


# Back-compat alias used by tests and legacy callers
def score(payload: Dict[str, Any], explain: bool = False):
    return infer_score(payload, explain=explain)


# ---- Ensemble scoring (logistic + rf + gb) ----------------------------------
def infer_score_ensemble(
    payload: Dict[str, Any],
    *,
    explain: bool = False,
    top_n: int = 5,
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Average probabilities across available models.
    - If only logistic exists, ensemble equals logistic.
    - Missing/failed models are ignored (NOT counted as 0%).
    - Band is min..max of included probabilities (0 width if single model).
    - Per-model votes are returned; for missing models we omit the key.
    """
    # Always try to get feature order from logistic meta first
    logi_model, logi_meta, _ = _load_logistic(models_dir)
    feats = payload.get("features", {}) or {}
    demo_mode = False

    if not logi_meta or logi_model is None:
        # Full demo: synthesize a probability from features (or 6.2% flat)
        base_p = _sigmoid(0.1 * float(feats.get("burst_z", 0.0))) if feats else 0.062
        votes: Dict[str, float] = {"logistic": float(base_p)}
        probs = list(votes.values())
        demo_mode = True
        return {
            "prob_trigger_next_6h": float(np.mean(probs)),
            "low": float(min(probs)),
            "high": float(max(probs)),
            "per_model": votes,
            "demo": True,
        }

    feat_order = logi_meta.get("feature_order") or []
    if not feats:
        # Keep behavior consistent with baseline when features are absent
        return {
            "prob_trigger_next_6h": 0.062,
            "low": 0.062,
            "high": 0.062,
            "per_model": {"logistic": 0.062},
            "note": "origin path not wired in infer",
            "demo": bool(logi_meta.get("demo", False)),
        }

    x = _vectorize(feats, feat_order)

    votes: Dict[str, float] = {}
    probs: List[float] = []

    # Logistic (required if we got here)
    try:
        p = float(logi_model.predict_proba(x)[0, 1])  # type: ignore[attr-defined]
        if np.isfinite(p):
            votes["logistic"] = p
            probs.append(p)
    except Exception:
        pass

    # Random Forest (optional)
    rf_model, _ = _load_rf(models_dir)
    if rf_model is not None:
        try:
            p = float(rf_model.predict_proba(x)[0, 1])  # type: ignore[attr-defined]
            if np.isfinite(p):
                votes["rf"] = p
                probs.append(p)
        except Exception:
            # ignore bad models
            pass

    # Gradient Boosting (optional)
    gb_model, _ = _load_gb(models_dir)
    if gb_model is not None:
        try:
            p = float(gb_model.predict_proba(x)[0, 1])  # type: ignore[attr-defined]
            if np.isfinite(p):
                votes["gb"] = p
                probs.append(p)
        except Exception:
            pass

    # Safety: ensure we have at least one probability
    if not probs:
        # Fall back to baseline demo transform
        base_p = _sigmoid(0.1 * float(feats.get("burst_z", 0.0)))
        votes = {"logistic": float(base_p)}
        probs = [float(base_p)]
        demo_mode = True

    mean_p = float(np.mean(probs))
    low_p = float(min(probs))
    high_p = float(max(probs))

    out: Dict[str, Any] = {
        "prob_trigger_next_6h": mean_p,
        "low": low_p,
        "high": high_p,
        "per_model": votes,
    }

    if explain and "logistic" in votes:
        # Provide linear contributions for logistic only.
        contrib = _contributions_linear(logi_model, x, feat_order, top_n)
        if contrib:
            out["contributions"] = contrib

    if demo_mode or bool(logi_meta.get("demo", False)):
        out["demo"] = True

    return out


# ---- Live backtest (used by CI summary) -------------------------------------
def _load_jsonl(path: Path) -> List[dict]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except Exception:
        return []


def _parse_ts(v) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        try:
            s = str(v)
            s = s[:-1] + "+00:00" if s.endswith("Z") else s
            return datetime.fromisoformat(s).astimezone(timezone.utc)
        except Exception:
            return None


def _label_has_trigger_between(triggers, origin: str, t0: datetime, t1: datetime) -> int:
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
    Score each origin at each hourly bucket over the last 24h and compute a simple
    precision/recall snapshot. Uses baseline `score()` (logistic) to stay cheap.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [now - timedelta(hours=i) for i in range(24, 0, -1)]

    flags = _load_jsonl(RETRAINING_LOG_PATH)
    origins = sorted({
        _norm(r.get("origin", "unknown"))
        for r in flags
        if _parse_ts(r.get("timestamp")) and _parse_ts(r.get("timestamp")) >= now - timedelta(hours=24)
    }) or ["twitter", "reddit", "rss_news"]

    triggers = _load_jsonl(RETRAINING_TRIGGERED_LOG_PATH)

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
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        per_origin.append({
            "origin": o,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })

    # overall
    tp = sum(po["tp"] for po in per_origin)
    fp = sum(po["fp"] for po in per_origin)
    fn = sum(po["fn"] for po in per_origin)
    prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0

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