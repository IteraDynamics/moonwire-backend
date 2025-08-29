from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
from datetime import datetime, timedelta, timezone

from src.paths import MODELS_DIR, RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH
from src.analytics.origin_utils import normalize_origin as _norm

# NEW: dynamic threshold helpers
from src.ml.recent_scores import append_recent_score, dynamic_threshold_for_origin

# Filenames
_LOGI_MODEL = "trigger_likelihood_v0.joblib"
_LOGI_META  = "trigger_likelihood_v0.meta.json"
_COV_NAME   = "feature_coverage.json"

_RF_MODEL   = "trigger_likelihood_rf.joblib"
_RF_META    = "trigger_likelihood_rf.meta.json"

_GB_MODEL   = "trigger_likelihood_gb.joblib"
_GB_META    = "trigger_likelihood_gb.meta.json"


# ---------- loaders ----------
def _artifact_paths(model_name: str, meta_name: str, models_dir: Path | None = None) -> Tuple[Path, Path]:
    md = models_dir or MODELS_DIR
    return md / model_name, md / meta_name

def _load_model_and_meta(model_name: str, meta_name: str, models_dir: Path | None = None):
    mpath, jpath = _artifact_paths(model_name, meta_name, models_dir)
    model = joblib.load(mpath)
    with jpath.open("r") as f:
        meta = json.load(f)
    return model, meta

def _load_cov(models_dir: Path | None = None) -> Dict[str, Any]:
    cpath = (models_dir or MODELS_DIR) / _COV_NAME
    if not cpath.exists():
        return {}
    try:
        with cpath.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------- public metadata helpers ----------
def model_metadata(models_dir: Path | None = None) -> Dict[str, Any]:
    """Logistic metadata (+ coverage merged) for back-compat."""
    try:
        _, meta = _load_model_and_meta(_LOGI_MODEL, _LOGI_META, models_dir)
        cov = _load_cov(models_dir)
        out = dict(meta)
        if cov:
            out["feature_coverage"] = cov
        return out
    except Exception:
        return {}

def model_metadata_all(models_dir: Path | None = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # logistic
    L = model_metadata(models_dir)
    if L:
        out["logistic"] = L
    # rf
    try:
        _, m = _load_model_and_meta(_RF_MODEL, _RF_META, models_dir)
        out["rf"] = m
    except Exception:
        pass
    # gb
    try:
        _, m = _load_model_and_meta(_GB_MODEL, _GB_META, models_dir)
        out["gb"] = m
    except Exception:
        pass
    return out


# ---------- scoring ----------
def _vectorize(features: Dict[str, Any], feat_order: List[str]) -> np.ndarray:
    return np.array([[float(features.get(k, 0.0) or 0.0) for k in feat_order]], dtype=float)

def _contributions_linear(model, xrow: np.ndarray, feat_order: List[str], top_n: int | None) -> Dict[str, float]:
    try:
        coef = model.coef_.ravel()
        contrib = {feat_order[i]: float(coef[i] * xrow[0, i]) for i in range(len(feat_order))}
        if top_n is not None:
            return dict(sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n])
        return contrib
    except Exception:
        return {}

def infer_score(payload: Dict[str, Any], *, explain: bool = False, top_n: int = 5, models_dir: Path | None = None) -> Dict[str, Any]:
    try:
        model, meta = _load_model_and_meta(_LOGI_MODEL, _LOGI_META, models_dir)
    except Exception:
        if payload.get("features"):
            bz = float(payload["features"].get("burst_z", 0.0))
            p = 1 / (1 + np.exp(-0.1 * bz))
            res = {"prob_trigger_next_6h": float(p), "demo": True}
            if explain:
                res["contributions"] = {"burst_z": 0.1 * bz}
            return res
        return {"prob_trigger_next_6h": 0.062, "demo": True}

    feats = payload.get("features")
    if feats is None:
        return {"prob_trigger_next_6h": 0.062, "note": "origin path not wired", "demo": meta.get("demo", False)}

    feat_order = meta.get("feature_order") or []
    x = _vectorize(feats, feat_order)
    proba = float(model.predict_proba(x)[0, 1])
    out = {"prob_trigger_next_6h": proba}
    if explain:
        out["contributions"] = _contributions_linear(model, x, feat_order, top_n=top_n)
    return out


def infer_score_ensemble(payload: Dict[str, Any], *, models_dir: Path | None = None) -> Dict[str, Any]:
    """
    Ensemble scoring over available models (logistic + random forest + gb).
    Returns:
      {
        "prob_trigger_next_6h": float,
        "low": float, "high": float,   # mean ± band (min/max),
        "votes": {"logistic": p1, "rf": p2, "gb": p3}, "models": [...],
        # NEW (non-breaking additions):
        "threshold_dynamic": float|None,
        "threshold_static": float,
        "threshold_used": "dynamic"|"static",
        "recent_count": int
      }
    Falls back to demo when needed.
    """
    votes: Dict[str, float] = {}
    demo = False

    feats = payload.get("features") or {}
    origin = str(payload.get("origin") or payload.get("source") or "unknown")

    # try logistic
    try:
        L_model, L_meta = _load_model_and_meta(_LOGI_MODEL, _LOGI_META, models_dir)
        order = L_meta.get("feature_order") or []
        votes["logistic"] = float(L_model.predict_proba(_vectorize(feats, order))[0, 1])
    except Exception:
        pass

    # try RF
    try:
        RF_model, RF_meta = _load_model_and_meta(_RF_MODEL, _RF_META, models_dir)
        order = RF_meta.get("feature_order") or []
        votes["rf"] = float(RF_model.predict_proba(_vectorize(feats, order))[0, 1])
    except Exception:
        pass

    # try GB
    try:
        GB_model, GB_meta = _load_model_and_meta(_GB_MODEL, _GB_META, models_dir)
        order = GB_meta.get("feature_order") or []
        votes["gb"] = float(GB_model.predict_proba(_vectorize(feats, order))[0, 1])
    except Exception:
        pass

    if not votes:
        bz = float(feats.get("burst_z", 0.0))
        p_demo = 1 / (1 + np.exp(-0.08 * bz))
        votes["logistic"] = p_demo
        demo = True

    probs = list(votes.values())
    mean = float(np.mean(probs))
    low = float(min(probs))
    high = float(max(probs))

    # --- NEW: log this score and compute dynamic/static thresholds
    try:
        append_recent_score(origin, mean)
    except Exception:
        pass

    dyn_thr, n_recent, static_thr = dynamic_threshold_for_origin(origin)
    used = "dynamic" if dyn_thr is not None else "static"
    thr_used = float(dyn_thr if dyn_thr is not None else static_thr)

    return {
        "prob_trigger_next_6h": mean,
        "low": low,
        "high": high,
        "votes": votes,
        "models": list(votes.keys()),
        "demo": demo,
        # new fields (harmless to downstream callers):
        "threshold_dynamic": dyn_thr,
        "threshold_static": static_thr,
        "threshold_used": used,
        "recent_count": n_recent,
        "decision": bool(mean >= thr_used),
    }


# Back-compat alias
def score(payload: dict, explain: bool = False):
    return infer_score(payload, explain=explain)

__all__ = ["infer_score", "infer_score_ensemble", "score", "model_metadata", "model_metadata_all"]


# ---------- Online backtest (kept from v0.2) ----------
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
            s = str(v); s = s[:-1] + "+00:00" if s.endswith("Z") else s
            return datetime.fromisoformat(s).astimezone(timezone.utc)
        except Exception:
            return None

def _label_has_trigger_between(triggers, origin: str, t0: datetime, t1: datetime) -> int:
    o = _norm(origin)
    for r in triggers:
        if _norm(r.get("origin","")) != o: continue
        ts = _parse_ts(r.get("timestamp"))
        if ts and t0 < ts <= t1:
            return 1
    return 0

def live_backtest_last_24h(interval: str = "hour", threshold: float = 0.5) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [now - timedelta(hours=i) for i in range(24, 0, -1)]
    flags = _load_jsonl(RETRAINING_LOG_PATH)
    origins = sorted({_norm(r.get("origin","unknown")) for r in flags if _parse_ts(r.get("timestamp")) and _parse_ts(r.get("timestamp")) >= now - timedelta(hours=24)}) or ["twitter","reddit","rss_news"]
    triggers = _load_jsonl(RETRAINING_TRIGGERED_LOG_PATH)

    per_origin = []
    for o in origins[:10]:
        tp=fp=fn=tn=0
        for t in buckets:
            t_iso = t.isoformat()
            try:
                p = score({"origin": o, "timestamp": t_iso}).get("prob_trigger_next_6h", 0.0)
            except Exception:
                p = 0.0
            y = _label_has_trigger_between(triggers, o, t, t + timedelta(hours=6))
            yhat = 1 if p >= threshold else 0
            if   yhat==1 and y==1: tp+=1
            elif yhat==1 and y==0: fp+=1
            elif yhat==0 and y==1: fn+=1
            else: tn+=1
        prec = tp/float(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/float(tp+fn) if (tp+fn)>0 else 0.0
        per_origin.append({"origin": o, "precision": round(prec,3), "recall": round(rec,3), "tp":tp,"fp":fp,"fn":fn,"tn":tn})

    tp=sum(po["tp"] for po in per_origin); fp=sum(po["fp"] for po in per_origin)
    fn=sum(po["fn"] for po in per_origin); tn=sum(po["tn"] for po in per_origin)
    prec = tp/float(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/float(tp+fn) if (tp+fn)>0 else 0.0

    return {
        "window_hours": 24,
        "threshold": threshold,
        "overall": {"precision": round(prec,3), "recall": round(rec,3), "tp":tp,"fp":fp,"fn":fn,"tn":tn},
        "per_origin": per_origin[:3],
    }
