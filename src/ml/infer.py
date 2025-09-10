# src/ml/infer.py
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
from datetime import datetime, timedelta, timezone

from src.paths import MODELS_DIR, RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH
from src.analytics.origin_utils import normalize_origin as _norm

# Filenames
_LOGI_MODEL = "trigger_likelihood_v0.joblib"
_LOGI_META  = "trigger_likelihood_v0.meta.json"
_COV_NAME   = "feature_coverage.json"

_RF_MODEL   = "trigger_likelihood_rf.joblib"
_RF_META    = "trigger_likelihood_rf.meta.json"

_GB_MODEL   = "trigger_likelihood_gb.joblib"
_GB_META    = "trigger_likelihood_gb.meta.json"

# Trigger history path (env overrideable)
def _resolve_trigger_log_path() -> Path:
    envp = os.getenv("TRIGGER_LOG_PATH")
    return Path(envp) if envp else (MODELS_DIR / "trigger_history.jsonl")

_TRIGGER_LOG_PATH = _resolve_trigger_log_path()

# Version file path
_VERSION_FILE = MODELS_DIR / "training_version.txt"


# ---------- helpers ----------
def _read_model_version() -> str:
    """
    Read current training/model version from models/training_version.txt.
    Fallback to 'unknown' if missing/unreadable.
    """
    try:
        if _VERSION_FILE.exists():
            txt = _VERSION_FILE.read_text().strip()
            return txt or "unknown"
    except Exception:
        pass
    return "unknown"


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
    version = _read_model_version()
    try:
        model, meta = _load_model_and_meta(_LOGI_MODEL, _LOGI_META, models_dir)
    except Exception:
        if payload.get("features"):
            bz = float(payload["features"].get("burst_z", 0.0))
            p = 1 / (1 + np.exp(-0.1 * bz))
            res = {"prob_trigger_next_6h": float(p), "demo": True, "model_version": version}
            if explain:
                res["contributions"] = {"burst_z": 0.1 * bz}
            return res
        return {"prob_trigger_next_6h": 0.062, "demo": True, "model_version": version}

    feats = payload.get("features")
    if feats is None:
        return {
            "prob_trigger_next_6h": 0.062,
            "note": "origin path not wired",
            "demo": meta.get("demo", False),
            "model_version": version,
        }

    feat_order = meta.get("feature_order") or []
    x = _vectorize(feats, feat_order)
    proba = float(model.predict_proba(x)[0, 1])
    out = {"prob_trigger_next_6h": proba, "model_version": version}
    if explain:
        out["contributions"] = _contributions_linear(model, x, feat_order, top_n=top_n)
    return out


def infer_score_ensemble(payload: Dict[str, Any], *, models_dir: Path | None = None) -> Dict[str, Any]:
    """
    Ensemble scoring over available models (logistic + random forest + gb).
    Returns dict with mean prob, band, votes, etc., plus model_version.
    """
    version = _read_model_version()
    votes: Dict[str, float] = {}
    demo = False
    feats = payload.get("features") or {}

    # logistic
    try:
        L_model, L_meta = _load_model_and_meta(_LOGI_MODEL, _LOGI_META, models_dir)
        order = L_meta.get("feature_order") or []
        votes["logistic"] = float(L_model.predict_proba(_vectorize(feats, order))[0, 1])
    except Exception:
        pass

    # RF
    try:
        RF_model, RF_meta = _load_model_and_meta(_RF_MODEL, _RF_META, models_dir)
        order = RF_meta.get("feature_order") or []
        votes["rf"] = float(RF_model.predict_proba(_vectorize(feats, order))[0, 1])
    except Exception:
        pass

    # GB
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

    out: Dict[str, Any] = {
        "prob_trigger_next_6h": mean,
        "low": low,
        "high": high,
        "votes": votes,
        "models": list(votes.keys()),
        "demo": demo,
        "model_version": version,
    }

    # --- Append to trigger history log (version-tagged) ---
    # We log the best info we have; if the caller later adds an explanation/threshold,
    # that gets logged elsewhere (e.g., via API or subsequent pass). This is a minimal,
    # safe history record so we don't lose events.
    try:
        # Ensure parent dir exists
        _TRIGGER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Allow upstream to pass decision/meta in payload (non-breaking)
        explanation = (payload or {}).get("explanation") or {}
        decision = explanation.get("decision") or ("triggered" if mean >= 0.5 else "not_triggered")
        threshold_used = explanation.get("threshold_after_volatility")
        drifted = explanation.get("drifted_features", [])
        top_contribs = explanation.get("top_contributors", [])
        regime = explanation.get("volatility_regime")

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "origin": (payload.get("origin") or "unknown") if isinstance(payload, dict) else "unknown",
            "adjusted_score": float(explanation.get("adjusted_score", mean)),
            "threshold": threshold_used,
            "decision": str(decision),
            "volatility_regime": regime,
            "drifted_features": drifted if isinstance(drifted, list) else [],
            "top_contributors": top_contribs if isinstance(top_contribs, list) else [],
            "model_version": version,
        }
        with _TRIGGER_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Never break inference due to logging problems
        pass

    return out


# ---------- Volatility-aware thresholds ----------
def compute_volatility_adjusted_threshold(base_threshold: float, regime: str) -> Dict[str, Any]:
    """
    Adjust threshold based on volatility regime.
    """
    try:
        mults = {
            "calm": float(os.getenv("TL_REGIME_THRESH_MULT_CALM", "0.9")),
            "normal": float(os.getenv("TL_REGIME_THRESH_MULT_NORMAL", "1.0")),
            "turbulent": float(os.getenv("TL_REGIME_THRESH_MULT_TURBULENT", "1.1")),
        }
        multiplier = mults.get(str(regime).strip().lower(), 1.0)
    except Exception:
        multiplier = 1.0

    adjusted = base_threshold * multiplier
    return {
        "base_threshold": base_threshold,
        "volatility_regime": regime,
        "regime_multiplier": multiplier,
        "threshold_after_volatility": adjusted,
    }


# Back-compat alias
def score(payload: dict, explain: bool = False):
    return infer_score(payload, explain=explain)

__all__ = [
    "infer_score",
    "infer_score_ensemble",
    "score",
    "model_metadata",
    "model_metadata_all",
    "compute_volatility_adjusted_threshold",
]


# ---------- Online backtest ----------
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
            try:
                p = score({"origin": o, "timestamp": t.isoformat()}).get("prob_trigger_next_6h", 0.0)
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
        per_origin.append({"origin": o, "precision": round(prec,3), "recall": round(rec,3),
                           "tp":tp,"fp":fp,"fn":fn,"tn":tn})

    tp=sum(po["tp"] for po in per_origin); fp=sum(po["fp"] for po in per_origin)
    fn=sum(po["fn"] for po in per_origin); tn=sum(po["tn"] for po in per_origin)
    prec = tp/float(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/float(tp+fn) if (tp+fn)>0 else 0.0

    return {
        "window_hours": 24,
        "threshold": threshold,
        "overall": {"precision": round(prec,3), "recall": round(rec,3),
                    "tp":tp,"fp":fp,"fn":fn,"tn":tn},
        "per_origin": per_origin[:3],
    }