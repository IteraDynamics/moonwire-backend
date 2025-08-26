# src/ml/infer.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from datetime import datetime, timedelta, timezone

from src.paths import MODELS_DIR, RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH
from src.analytics.origin_utils import normalize_origin as _norm

LR_NAME = "trigger_likelihood_v0"
RF_NAME = "trigger_likelihood_rf"
GB_NAME = "trigger_likelihood_gb"

_META = ".meta.json"
_MODEL = ".joblib"
_COV_NAME = "feature_coverage.json"


# --------- artifact helpers ----------
def _paths_for(model_stub: str, models_dir: Path | None = None) -> Tuple[Path, Path]:
    md = models_dir or MODELS_DIR
    return md / f"{model_stub}{_MODEL}", md / f"{model_stub}{_META}"


def _load_model_and_meta(model_stub: str, models_dir: Path | None = None):
    mpath, jpath = _paths_for(model_stub, models_dir)
    model = joblib.load(mpath)
    with jpath.open("r") as f:
        meta = json.load(f)
    # Optional shared coverage file
    cov_path = (models_dir or MODELS_DIR) / _COV_NAME
    cov = {}
    if cov_path.exists():
        try:
            with cov_path.open("r") as f:
                cov = json.load(f)
        except Exception:
            cov = {}
    if cov:
        meta = dict(meta)
        meta["feature_coverage"] = cov
    return model, meta


def _vectorize(features: Dict[str, Any], feat_order: List[str]) -> np.ndarray:
    return np.array([[float(features.get(k, 0.0) or 0.0) for k in feat_order]], dtype=float)


# --------- public metadata helpers ----------
def model_metadata(models_dir: Path | None = None) -> Dict[str, Any]:
    """Back-compat: return logistic meta if present; else {}."""
    try:
        _, meta = _load_model_and_meta(LR_NAME, models_dir)
        return meta
    except Exception:
        return {}


def model_metadata_all(models_dir: Path | None = None) -> Dict[str, Any]:
    """Return nested metadata blocks for whichever models exist."""
    out: Dict[str, Any] = {}
    for stub, key in ((LR_NAME, "logistic"), (RF_NAME, "rf"), (GB_NAME, "gb")):
        try:
            _, meta = _load_model_and_meta(stub, models_dir)
            out[key] = meta
        except Exception:
            pass
    # keep tests happy: expose top-level "metrics" if logistic is present
    if "logistic" in out and "metrics" in out["logistic"]:
        out["metrics"] = out["logistic"]["metrics"]
    return out


# --------- scoring ----------
def _predict_with(model, xrow: np.ndarray) -> float:
    return float(model.predict_proba(xrow)[0, 1])


def infer_score(payload: Dict[str, Any], *, explain: bool = False, top_n: int = 5, models_dir: Path | None = None) -> Dict[str, Any]:
    """Logistic-only (for back-compat)."""
    try:
        lr, meta = _load_model_and_meta(LR_NAME, models_dir)
    except Exception:
        # minimal demo fallback
        feats = payload.get("features", {}) or {}
        p = 1 / (1 + np.exp(-0.1 * float(feats.get("burst_z", 0.0))))
        out = {"prob_trigger_next_6h": float(p), "demo": True}
        return out

    feat_order = meta.get("feature_order") or []
    feats = payload.get("features") or {}
    x = _vectorize(feats, feat_order)
    proba = _predict_with(lr, x)
    return {"prob_trigger_next_6h": proba}


def infer_score_ensemble(payload: Dict[str, Any], *, models_dir: Path | None = None) -> Dict[str, Any]:
    """
    Try logistic, rf, gb; average available votes.
    Confidence band: min..max of available votes (degenerates to ±0 when one model).
    """
    votes: Dict[str, float] = {}
    feat_order = None

    # load each model if present
    for stub, key in ((LR_NAME, "logistic"), (RF_NAME, "rf"), (GB_NAME, "gb")):
        try:
            model, meta = _load_model_and_meta(stub, models_dir)
            feat_order = feat_order or (meta.get("feature_order") or [])
            feats = payload.get("features") or {}
            x = _vectorize(feats, feat_order)
            votes[key] = _predict_with(model, x)
        except Exception:
            pass

    if not votes:
        # demo fallback
        feats = payload.get("features", {}) or {}
        p = 1 / (1 + np.exp(-0.1 * float(feats.get("burst_z", 0.0))))
        return {
            "prob_trigger_next_6h": float(p),
            "lo": float(p),
            "hi": float(p),
            "votes": {"logistic": float(p)},
            "demo": True,
        }

    probs = list(votes.values())
    mean = float(sum(probs) / len(probs))
    lo = float(min(probs))
    hi = float(max(probs))
    return {"prob_trigger_next_6h": mean, "lo": lo, "hi": hi, "votes": votes}


# --- Back-compat shim expected by tests
def score(payload: dict, explain: bool = False):
    return infer_score(payload, explain=explain)


__all__ = [
    "infer_score",
    "infer_score_ensemble",
    "score",
    "model_metadata",
    "model_metadata_all",
]


# --------- (unchanged) live backtest utilities ----------
def _load_jsonl(path) -> List[dict]:
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
    origins = sorted({ _norm(r.get("origin","unknown")) for r in flags if _parse_ts(r.get("timestamp")) and _parse_ts(r.get("timestamp")) >= now - timedelta(hours=24) }) or ["twitter","reddit","rss_news"]
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
