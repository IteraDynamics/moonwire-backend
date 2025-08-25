# src/ml/infer.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
from datetime import datetime, timedelta, timezone

# IMPORTANT: import the module, not names, so monkeypatching works
import src.paths as paths
from src.analytics.origin_utils import normalize_origin as _norm

# Filenames for logistic artifacts
_MODEL_NAME = "trigger_likelihood_v0.joblib"
_META_NAME  = "trigger_likelihood_v0.meta.json"
_COV_NAME   = "feature_coverage.json"

# (Optional) ensemble meta filename – used by the ensemble helper below
_ENSEMBLE_META = "trigger_likelihood_ensemble.meta.json"


# ---------------------- artifact helpers ----------------------
def _artifact_paths(models_dir: Path | None = None) -> Tuple[Path, Path, Path]:
    """Return (model_path, meta_path, coverage_path) for the logistic baseline."""
    md = models_dir or paths.MODELS_DIR
    return md / _MODEL_NAME, md / _META_NAME, md / _COV_NAME


def _load_model_and_meta(models_dir: Path | None = None):
    """Load logistic model + meta (+ optional coverage)."""
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


# ---------------------- scoring utilities ----------------------
def _vectorize(features: Dict[str, Any], feat_order: List[str]) -> np.ndarray:
    return np.array([[float(features.get(k, 0.0) or 0.0) for k in feat_order]], dtype=float)


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


# ---------------------- main logistic scoring ----------------------
def infer_score(
    payload: Dict[str, Any],
    *,
    explain: bool = False,
    top_n: int = 5,
    models_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Score either:
      A) {"origin": "...", "timestamp": "..."}  -> (kept as safe fallback)
      B) {"features": {...}}                    -> direct features dict (what CI summary uses)
    """
    try:
        model, meta, _ = _load_model_and_meta(models_dir)
    except Exception:
        # Demo fallback when no artifacts present
        if payload.get("features"):
            feats = payload["features"]
            # Simple synthetic mapping for demo
            p = 1 / (1 + np.exp(-0.1 * float(feats.get("burst_z", 0.0))))
            demo_res = {"prob_trigger_next_6h": float(p), "demo": True}
            if explain:
                demo_res["contributions"] = {"burst_z": float(0.1 * feats.get("burst_z", 0.0))}
            return demo_res
        return {"prob_trigger_next_6h": 0.062, "demo": True}

    feat_order = meta.get("feature_order") or []

    feats = payload.get("features")
    if feats is None:
        # Maintain compatibility: if only origin/timestamp supplied, return a safe value
        return {
            "prob_trigger_next_6h": 0.062,
            "note": "origin path not wired in infer",
            "demo": bool(meta.get("demo", False)),
        }

    x = _vectorize(feats, feat_order)
    proba = float(model.predict_proba(x)[0, 1])

    out = {"prob_trigger_next_6h": proba}
    if explain:
        out["contributions"] = _contributions(model, x, feat_order, top_n=top_n)
    return out


# Back-compat shim for tests & callers expecting `score`
def score(payload: dict, explain: bool = False):
    """Backward-compatible alias for infer_score."""
    return infer_score(payload, explain=explain)


# ---------------------- ensemble scoring (safe fallback) ----------------------
def infer_score_ensemble(
    payload: Dict[str, Any],
    *,
    explain: bool = False,
    top_n: int = 3,
    models_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Try to score via ensemble artifacts; if not present, fall back to logistic.
    Returns:
      {
        "prob_trigger_next_6h": float,
        "low": float|None,
        "high": float|None,
        "votes": {"logistic": p, "rf": p, "gb": p, ...}  # when available
      }
    """
    md = models_dir or paths.MODELS_DIR
    ens_meta_path = md / _ENSEMBLE_META

    # If ensemble meta exists, try to use it (minimal reader).
    # Otherwise, fall back to logistic scoring.
    try:
        if ens_meta_path.exists():
            with ens_meta_path.open("r") as f:
                emeta = json.load(f)
            feat_order = emeta.get("feature_order") or []
            feats = payload.get("features")
            if feats is None:
                # Keep the same safe default as logistic
                return {"prob_trigger_next_6h": 0.062, "low": None, "high": None, "note": "origin path not wired"}

            x = _vectorize(feats, feat_order)

            votes: Dict[str, float] = {}
            probs: List[float] = []

            # Attempt to load each listed model; ignore any that fail
            for name, rel_path in (emeta.get("models") or {}).items():
                try:
                    model = joblib.load(md / rel_path)
                    p = float(model.predict_proba(x)[0, 1])
                    votes[name] = p
                    probs.append(p)
                except Exception:
                    continue

            if probs:
                p_mean = float(np.mean(probs))
                # Confidence band from bootstrap std saved in meta (if present)
                p_std = float(emeta.get("pred_std", 0.0) or 0.0)
                low = max(0.0, p_mean - p_std)
                high = min(1.0, p_mean + p_std)
                out = {"prob_trigger_next_6h": p_mean, "low": low, "high": high, "votes": votes}
                if explain:
                    # No unified per-feature contributions across different model types;
                    # rely on logistic explain if available.
                    out["note"] = out.get("note", "ensemble explain not available; using averaged probabilities")
                return out
        # Fall through to logistic
    except Exception:
        pass

    # Fallback: logistic-only
    base = infer_score(payload, explain=explain, top_n=top_n, models_dir=models_dir)
    base.setdefault("low", None)
    base.setdefault("high", None)
    base.setdefault("votes", {"logistic": base.get("prob_trigger_next_6h")})
    return base


# ---------------------- Online inference / live backtest ----------------------
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
    """Score each origin at each hourly bucket over the last 24h and compute precision/recall snapshot."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [now - timedelta(hours=i) for i in range(24, 0, -1)]  # 24 exclusive to now

    flags = _load_jsonl(paths.RETRAINING_LOG_PATH)
    triggers = _load_jsonl(paths.RETRAINING_TRIGGERED_LOG_PATH)

    # Get origins seen in last 24h of flags; if none, use a demo set
    origins = sorted(
        {
            _norm(r.get("origin", "unknown"))
            for r in flags
            if (ts := _parse_ts(r.get("timestamp"))) and ts >= now - timedelta(hours=24)
        }
    ) or ["twitter", "reddit", "rss_news"]

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
        per_origin.append(
            {"origin": o, "precision": round(prec, 3), "recall": round(rec, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        )

    # overall
    tp = sum(po["tp"] for po in per_origin)
    fp = sum(po["fp"] for po in per_origin)
    fn = sum(po["fn"] for po in per_origin)
    # tn = sum(po["tn"] for po in per_origin)  # not used in summary
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