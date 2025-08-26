# src/ml/infer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from datetime import datetime, timedelta, timezone

from src.paths import (
    MODELS_DIR,
    RETRAINING_LOG_PATH,
    RETRAINING_TRIGGERED_LOG_PATH,
)
from src.analytics.origin_utils import normalize_origin as _norm


# ---------------------------------------------------------------------------
# Artifact names
# ---------------------------------------------------------------------------
_LOG_JOBLIB = "trigger_likelihood_v0.joblib"
_LOG_META   = "trigger_likelihood_v0.meta.json"
_RF_JOBLIB  = "trigger_likelihood_rf.joblib"
_RF_META    = "trigger_likelihood_rf.meta.json"
_GB_JOBLIB  = "trigger_likelihood_gb.joblib"
_GB_META    = "trigger_likelihood_gb.meta.json"
_COV_JSON   = "feature_coverage.json"


def _paths(models_dir: Path | None = None) -> Dict[str, Path]:
    md = models_dir or MODELS_DIR
    return {
        "log_model": md / _LOG_JOBLIB,
        "log_meta":  md / _LOG_META,
        "rf_model":  md / _RF_JOBLIB,
        "rf_meta":   md / _RF_META,
        "gb_model":  md / _GB_JOBLIB,
        "gb_meta":   md / _GB_META,
        "coverage":  md / _COV_JSON,
    }


def _load_json(p: Path) -> Dict[str, Any]:
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_model(p: Path):
    try:
        return joblib.load(p)
    except Exception:
        return None


def _vectorize(features: Dict[str, Any], feat_order: List[str]) -> np.ndarray:
    return np.array([[float(features.get(k, 0.0) or 0.0) for k in feat_order]], dtype=float)


def _contributions_linear(model, xrow: np.ndarray, feat_order: List[str], top_n: int | None) -> Dict[str, float]:
    """
    Linear contributions: coef * value for logistic models.
    Sorted by |contribution| if top_n is provided.
    """
    try:
        coef = model.coef_.ravel()
    except Exception:
        return {}

    contrib = {k: float(coef[i] * xrow[0, i]) for i, k in enumerate(feat_order)}
    if top_n is not None:
        items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
        return {k: float(v) for k, v in items}
    return contrib


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def model_metadata(models_dir: Path | None = None) -> Dict[str, Any]:
    """
    Returns metadata with both a backward-compatible top-level logistic block
    AND nested blocks for each available model: logistic, rf, gb.
    """
    p = _paths(models_dir)

    # Load coverage (optional)
    coverage = _load_json(p["coverage"])

    # Load per-model metas
    log_meta = _load_json(p["log_meta"])
    rf_meta  = _load_json(p["rf_meta"])
    gb_meta  = _load_json(p["gb_meta"])

    # Attach coverage reference into logistic meta for convenience
    if log_meta:
        if coverage:
            log_meta.setdefault("feature_coverage", coverage)
        # For some callers/tests that expect top-level fields:
        top = {
            "created_at": log_meta.get("created_at"),
            "git_sha": log_meta.get("git_sha"),
            "feature_order": log_meta.get("feature_order"),
            "metrics": log_meta.get("metrics"),
            "demo": bool(log_meta.get("demo", False)),
            "artifacts": log_meta.get("artifacts", {}),
            "top_features": log_meta.get("top_features", []),
            "feature_coverage_summary": log_meta.get("feature_coverage_summary", {}),
        }
    else:
        top = {}

    out: Dict[str, Any] = dict(top)
    if log_meta:
        out["logistic"] = log_meta
    if rf_meta:
        out["rf"] = rf_meta
    if gb_meta:
        out["gb"] = gb_meta
    if coverage and "feature_coverage" not in out:
        out["feature_coverage"] = coverage

    return out


# ---------------------------------------------------------------------------
# Logistic scorer (back-compat)
# ---------------------------------------------------------------------------
def infer_score(
    payload: Dict[str, Any],
    *,
    explain: bool = False,
    top_n: int = 5,
    models_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Score using the logistic model. If artifacts are missing, returns a safe demo value.
    """
    p = _paths(models_dir)
    feats = payload.get("features") or {}

    log_model = _load_model(p["log_model"])
    log_meta  = _load_json(p["log_meta"])

    # Demo fallback when logistic artifacts are unavailable
    if log_model is None or not log_meta:
        bz = float(feats.get("burst_z", 0.0) or 0.0)
        demo_p = 1.0 / (1.0 + np.exp(-0.1 * bz))
        out = {"prob_trigger_next_6h": float(demo_p), "demo": True}
        if explain and "burst_z" in feats:
            out["contributions"] = {"burst_z": float(0.1 * bz)}
        return out

    feat_order = log_meta.get("feature_order") or []
    if not feat_order:
        # If meta is malformed, return a safe default
        return {"prob_trigger_next_6h": 0.062, "note": "no feature_order in meta", "demo": bool(log_meta.get("demo", False))}

    x = _vectorize(feats, feat_order)
    try:
        proba = float(log_model.predict_proba(x)[0, 1])
    except Exception:
        proba = 0.062

    out = {"prob_trigger_next_6h": proba}
    if explain:
        out["contributions"] = _contributions_linear(log_model, x, feat_order, top_n=top_n)
    return out


# Back-compat alias used by tests/callers
def score(payload: Dict[str, Any], explain: bool = False):
    return infer_score(payload, explain=explain)


# ---------------------------------------------------------------------------
# Ensemble scorer: logistic + rf + gb (only average models that work)
# ---------------------------------------------------------------------------
def _predict_proba_or_none(model, xrow: np.ndarray) -> float | None:
    try:
        return float(model.predict_proba(xrow)[0, 1])
    except Exception:
        return None


def infer_score_ensemble(
    payload: Dict[str, Any],
    *,
    top_n: int = 5,
    models_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Average probabilities across the subset of models that successfully load and score.
    Returns mean prob, low/high band (min/max across available votes), and per-model votes.
    Falls back to a demo rule if no model can score.
    """
    p = _paths(models_dir)
    feats = payload.get("features") or {}

    # Determine feature order from the first available meta (prefer logistic)
    feat_order: List[str] = []
    for meta_path in (p["log_meta"], p["rf_meta"], p["gb_meta"]):
        m = _load_json(meta_path)
        if m.get("feature_order"):
            feat_order = list(m["feature_order"])
            break

    x = _vectorize(feats, feat_order) if feat_order else np.array([[0.0]])

    votes: Dict[str, float] = {}
    vals: List[float] = []

    # Logistic
    log_model = _load_model(p["log_model"])
    if log_model is not None and feat_order:
        v = _predict_proba_or_none(log_model, x)
        if v is not None:
            votes["logistic"] = v
            vals.append(v)

    # Random Forest
    rf_model = _load_model(p["rf_model"])
    if rf_model is not None and feat_order:
        v = _predict_proba_or_none(rf_model, x)
        if v is not None:
            votes["rf"] = v
            vals.append(v)

    # Gradient Boosting
    gb_model = _load_model(p["gb_model"])
    if gb_model is not None and feat_order:
        v = _predict_proba_or_none(gb_model, x)
        if v is not None:
            votes["gb"] = v
            vals.append(v)

    # Demo fallback when nothing produced a vote
    demo = False
    if not vals:
        demo = True
        bz = float(feats.get("burst_z", 0.0) or 0.0)
        demo_p = 1.0 / (1.0 + np.exp(-0.1 * bz))
        vals = [demo_p]
        votes = {"logistic": demo_p}  # show one synthetic vote for clarity

    mean = float(np.mean(vals))
    lo   = float(np.min(vals))
    hi   = float(np.max(vals))

    return {
        "prob_trigger_next_6h": mean,
        "low": lo,
        "high": hi,
        "votes": votes,
        "demo": demo,
    }


# ---------------------------------------------------------------------------
# Live backtest (used in CI summary)
# ---------------------------------------------------------------------------
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
    Uses the back-compat `score()` (logistic) so it stays stable regardless of ensemble presence.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [now - timedelta(hours=i) for i in range(24, 0, -1)]  # 24 exclusive to now

    flags = _load_jsonl(RETRAINING_LOG_PATH)
    origins = sorted({
        _norm(r.get("origin", "unknown"))
        for r in flags
        if _parse_ts(r.get("timestamp")) and _parse_ts(r.get("timestamp")) >= now - timedelta(hours=24)
    }) or ["twitter", "reddit", "rss_news"]

    triggers = _load_jsonl(RETRAINING_TRIGGERED_LOG_PATH)

    per_origin = []
    for o in origins[:10]:  # cap just in case
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

    tp = sum(po["tp"] for po in per_origin)
    fp = sum(po["fp"] for po in per_origin)
    fn = sum(po["fn"] for po in per_origin)
    tn = sum(po["tn"] for po in per_origin)
    prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / float(tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "window_hours": 24,
        "threshold": threshold,
        "overall": {"precision": round(prec, 3), "recall": round(rec, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "per_origin": per_origin[:3],
    }


__all__ = [
    "infer_score",
    "score",
    "infer_score_ensemble",
    "model_metadata",
    "live_backtest_last_24h",
]