# src/ml/train_trigger_model.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

from src import paths
from src.ml.feature_builder import build_examples, synth_demo_dataset


@dataclass
class TrainArtifacts:
    model_path: Path
    meta_path: Path


def _mk_arrays(rows: List[Dict[str, Any]], feat_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    for r in rows:
        f = r.get("features", {}) or {}
        X.append([float(f.get(k, 0.0) or 0.0) for k in feat_order])
        y.append(int(r.get("label", 0)))
    return np.array(X, dtype=float), np.array(y, dtype=int)


def _coverage_from_X(X: np.ndarray, feat_order: List[str]) -> Dict[str, Dict[str, float]]:
    n = float(X.shape[0]) if X.size else 1.0
    out: Dict[str, Dict[str, float]] = {}
    if X.size == 0:
        for k in feat_order:
            out[k] = {"nonzero_pct": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return out
    for i, k in enumerate(feat_order):
        col = X[:, i]
        nonzero = float(np.count_nonzero(col)) / n * 100.0
        out[k] = {
            "nonzero_pct": float(nonzero),
            "mean": float(np.mean(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return out


def _safe_metric(fn, y_true, y_pred, default: float = 0.0) -> float:
    try:
        return float(fn(y_true, y_pred))
    except Exception:
        return default


def _metrics_for_probs(ytr, p_tr, yva, p_va) -> Dict[str, float]:
    return {
        "roc_auc_tr": _safe_metric(roc_auc_score, ytr, p_tr, 0.5),
        "roc_auc_va": _safe_metric(roc_auc_score, yva, p_va, 0.5),
        "pr_auc_tr": _safe_metric(average_precision_score, ytr, p_tr, 0.0),
        "pr_auc_va": _safe_metric(average_precision_score, yva, p_va, 0.0),
        "logloss_tr": _safe_metric(log_loss, ytr, p_tr, 0.0),
        "logloss_va": _safe_metric(log_loss, yva, p_va, 0.0),
        "brier_tr": _safe_metric(brier_score_loss, ytr, p_tr, 0.0),
        "brier_va": _safe_metric(brier_score_loss, yva, p_va, 0.0),
    }


def _top_coefficients(model: LogisticRegression, feat_order: List[str], top: int = 5) -> List[Dict[str, float]]:
    if not hasattr(model, "coef_") or model.coef_ is None:
        return []
    coef = model.coef_.ravel().tolist()
    pairs = list(zip(feat_order, coef))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "coef": float(v)} for k, v in pairs[:top]]


def _git_sha() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def train(days: int = 14, interval: str = "hour", out_dir: Path | None = None) -> Dict[str, Any]:
    """
    Trains logistic + random forest (when data exists or DEMO_MODE is on).
    Persists:
      - models/trigger_likelihood_v0.joblib + .meta.json (+ feature_coverage.json once)
      - models/trigger_likelihood_rf.joblib + .meta.json
    Returns a small dict with paths/metrics flags.
    """
    out_dir = out_dir or paths.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, feat_order = build_examples(
        paths.RETRAINING_LOG_PATH,
        paths.RETRAINING_TRIGGERED_LOG_PATH,
        days=days,
        interval=interval,
    )

    demo_used = False
    if not rows and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        rows, feat_order, _ = synth_demo_dataset()
        demo_used = True

    if not rows:
        raise RuntimeError("No training rows and DEMO_MODE is false; cannot train.")

    # Arrays + coverage
    X, y = _mk_arrays(rows, feat_order)
    coverage = _coverage_from_X(X, feat_order)

    # time-aware split (80/20)
    n = X.shape[0]
    cut = max(1, int(0.8 * n))
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = X[cut:], y[cut:] if n > 1 else (X[:], y[:])

    # ---- Train Logistic
    logi = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
    )
    logi.fit(Xtr, ytr)

    p_tr = logi.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va = logi.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    metrics_logi = _metrics_for_probs(ytr, p_tr, yva, p_va)

    model_path_logi = out_dir / "trigger_likelihood_v0.joblib"
    meta_path_logi = out_dir / "trigger_likelihood_v0.meta.json"
    cov_path = out_dir / "feature_coverage.json"

    joblib.dump(logi, model_path_logi)

    with cov_path.open("w") as f:
        json.dump(coverage, f, indent=2)

    meta_logi: Dict[str, Any] = {
        "model_type": "logistic_regression",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": metrics_logi,
        "demo": bool(demo_used),
        "artifacts": {
            "model": str(model_path_logi),
            "feature_coverage": str(cov_path),
        },
        "top_features": _top_coefficients(logi, feat_order, top=5),
        "feature_coverage_summary": {k: round(v.get("nonzero_pct", 0.0), 2) for k, v in coverage.items()},
    }

    with meta_path_logi.open("w") as f:
        json.dump(meta_logi, f, indent=2)

    # ---- Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(Xtr, ytr)

    p_tr_rf = rf.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va_rf = rf.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    metrics_rf = _metrics_for_probs(ytr, p_tr_rf, yva, p_va_rf)

    model_path_rf = out_dir / "trigger_likelihood_rf.joblib"
    meta_path_rf = out_dir / "trigger_likelihood_rf.meta.json"

    joblib.dump(rf, model_path_rf)

    meta_rf: Dict[str, Any] = {
        "model_type": "random_forest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": metrics_rf,
        "demo": bool(demo_used),
        "artifacts": {
            "model": str(model_path_rf),
            "feature_coverage": str(cov_path),
        },
        # RF has no linear coefs; omit top_features
    }
    with meta_path_rf.open("w") as f:
        json.dump(meta_rf, f, indent=2)

    return {
        "logistic": {"model_path": str(model_path_logi), "meta_path": str(meta_path_logi), "metrics": metrics_logi},
        "rf": {"model_path": str(model_path_rf), "meta_path": str(meta_path_rf), "metrics": metrics_rf},
        "coverage_path": str(cov_path),
        "demo": demo_used,
    }


if __name__ == "__main__":
    # minimal CLI compatibility for CI
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--interval", type=str, default="hour")
    args = ap.parse_args()

    train(days=args.days, interval=args.interval)