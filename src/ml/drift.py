# src/ml/drift.py
from __future__ import annotations
import json, math, os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.paths import LOGS_DIR, MODELS_DIR, RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH
from src.ml.feature_builder import build_examples

_EPS = 1e-9

def _bern_kl(p: float, q: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    q = min(max(q, _EPS), 1.0 - _EPS)
    return float(p*math.log(p/q) + (1.0-p)*math.log((1.0-p)/(1.0-q)))

def _rows_iter_features(rows, feat_order: List[str]) -> List[Dict[str, float]]:
    out = []
    for r in rows:
        # FeatureRow dataclass or dict fallback
        feats = getattr(r, "x", None)
        if feats is not None:
            out.append({k: float(feats[i]) for i, k in enumerate(feat_order)})
        else:
            out.append({k: float((r.get("features", {}) or {}).get(k, 0.0) or 0.0) for k in feat_order})
    return out

def _stats_from_rows(rows, feat_order: List[str]) -> Dict[str, Dict[str, float]]:
    feats_list = _rows_iter_features(rows, feat_order)
    n = float(len(feats_list)) if feats_list else 1.0
    sums = {k: 0.0 for k in feat_order}
    nz   = {k: 0.0 for k in feat_order}
    for f in feats_list:
        for k, v in f.items():
            sums[k] += v
            if v != 0.0: nz[k] += 1.0
    out = {}
    for k in feat_order:
        out[k] = {
            "mean": float(sums[k]/n),
            "nonzero_pct": float(100.0 * nz[k]/n)
        }
    return out

def _load_training_coverage(meta_path: Path | None = None) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    meta_path = meta_path or (MODELS_DIR / "trigger_likelihood_v0.meta.json")
    meta = {}
    cov = {}
    feat_order: List[str] = []
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return meta, cov, feat_order
    try:
        feat_order = list(meta.get("feature_order", []))
    except Exception:
        feat_order = []
    # Prefer explicit coverage artifact if present
    cov_path = None
    try:
        cov_path = Path(meta.get("artifacts", {}).get("feature_coverage", ""))
    except Exception:
        cov_path = None
    if cov_path and cov_path.exists():
        cov = json.loads(cov_path.read_text())
    else:
        # fallback: compact summary in meta (just nonzero_pct)
        cov = {k: {"nonzero_pct": float(v), "mean": 0.0} for k, v in (meta.get("feature_coverage_summary") or {}).items()}
    return meta, cov, feat_order

def compute_recent_stats(hours: int = 24, interval: str = "hour") -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, float]], List[str]]:
    """Build examples and aggregate feature stats over the last N hours."""
    rows, feat_order = build_examples(
        RETRAINING_LOG_PATH,
        RETRAINING_TRIGGERED_LOG_PATH,
        days=14,                # reuse builder window; we filter to last N buckets below
        interval=interval,
    )
    # Filter to last N hours by timestamp on rows
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    def _ts(r):
        t = getattr(r, "ts", None)
        if t is not None: return t
        return r.get("timestamp")  # type: ignore[attr-defined]
    recent = [r for r in rows if (_ts(r) and _ts(r) >= cutoff)]
    stats = _stats_from_rows(recent, feat_order) if recent else {k: {"mean":0.0,"nonzero_pct":0.0} for k in feat_order}
    meta, cov, feat_order = _load_training_coverage()
    return meta, cov, stats, feat_order

def compute_drift(hours: int = 24, interval: str = "hour", top: int = 3) -> Dict[str, Any]:
    """Compare recent feature stats vs training coverage. Returns ranked drift list."""
    meta, cov, stats, feat_order = compute_recent_stats(hours=hours, interval=interval)
    out_features: List[Dict[str, Any]] = []
    for k in feat_order:
        tr = cov.get(k, {})
        tr_mean = float(tr.get("mean", 0.0) or 0.0)
        tr_nz   = float(tr.get("nonzero_pct", 0.0) or 0.0) / 100.0
        tr_min  = float(tr.get("min", 0.0) or 0.0)
        tr_max  = float(tr.get("max", 0.0) or 0.0)

        rc = stats.get(k, {"mean":0.0,"nonzero_pct":0.0})
        rc_mean = float(rc.get("mean", 0.0) or 0.0)
        rc_nz   = float(rc.get("nonzero_pct", 0.0) or 0.0) / 100.0

        mean_delta = rc_mean - tr_mean
        # normalize by range if available
        denom = abs(tr_max - tr_min)
        mean_delta_norm = float(abs(mean_delta) / (denom if denom > 1e-6 else (abs(tr_mean) + 1.0)))
        kl = _bern_kl(rc_nz, tr_nz) if (0.0 < rc_nz < 1.0 or 0.0 < tr_nz < 1.0) else 0.0
        score = float(mean_delta_norm + 0.5*abs(kl))

        out_features.append({
            "feature": k,
            "mean_train": tr_mean, "mean_recent": rc_mean,
            "p_train_nonzero": tr_nz, "p_recent_nonzero": rc_nz,
            "kl_nonzero": float(kl),
            "delta_mean": float(mean_delta),
            "delta_mean_norm": float(mean_delta_norm),
            "score": score,
        })

    out_features.sort(key=lambda d: d["score"], reverse=True)
    # DEMO fallback
    if not out_features and os.getenv("DEMO_MODE","false").lower() in ("1","true","yes"):
        out_features = [
            {"feature":"count_6h","mean_train":1.0,"mean_recent":4.5,"p_train_nonzero":0.2,"p_recent_nonzero":0.6,"kl_nonzero":0.4,"delta_mean":3.5,"delta_mean_norm":1.2,"score":1.4},
            {"feature":"burst_z","mean_train":0.1,"mean_recent":1.8,"p_train_nonzero":0.05,"p_recent_nonzero":0.4,"kl_nonzero":0.6,"delta_mean":1.7,"delta_mean_norm":1.1,"score":1.4},
            {"feature":"count_24h","mean_train":5.0,"mean_recent":12.0,"p_train_nonzero":0.5,"p_recent_nonzero":0.9,"kl_nonzero":0.5,"delta_mean":7.0,"delta_mean_norm":1.0,"score":1.25},
        ]

    return {
        "window_hours": hours,
        "interval": interval,
        "top": top,
        "features": out_features[:top],
        "meta_used": bool(meta),
    }

# --- helpers for tests (allow synthetic stats without IO) ---
def compute_drift_from_stats(train_cov: Dict[str, Dict[str,float]],
                             recent_stats: Dict[str, Dict[str,float]],
                             feat_order: List[str], top:int=3) -> List[Dict[str,Any]]:
    rows = []
    for k in feat_order:
        tr = train_cov.get(k, {"mean":0.0,"nonzero_pct":0.0,"min":0.0,"max":0.0})
        rc = recent_stats.get(k, {"mean":0.0,"nonzero_pct":0.0})
        denom = abs(tr.get("max",0.0)-tr.get("min",0.0)) or (abs(tr.get("mean",0.0))+1.0)
        delta = float(rc.get("mean",0.0)-tr.get("mean",0.0))
        kl = _bern_kl(float(rc.get("nonzero_pct",0.0)/100.0), float(tr.get("nonzero_pct",0.0)/100.0))
        score = abs(delta)/denom + 0.5*abs(kl)
        rows.append({"feature":k,"score":score})
    rows.sort(key=lambda d:d["score"], reverse=True)
    return rows[:top]