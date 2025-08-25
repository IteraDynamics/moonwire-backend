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
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

import src.paths as paths
from src.ml.feature_builder import build_examples, synth_demo_dataset


@dataclass
class TrainOutputs:
    model_path: Path
    meta_path: Path
    coverage_path: Path
    metrics: Dict[str, Any]
    feature_order: List[str]
    demo: bool


def _mk_arrays(rows: List[Any], feat_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accepts either dict rows like {"features": {...}, "label": 0/1}
    or FeatureRow dataclass with .x (feature list) and .y (label).
    """
    X: List[List[float]] = []
    y: List[int] = []
    for r in rows:
        if hasattr(r, "x") and hasattr(r, "y"):  # dataclass path
            X.append([float(v) for v in r.x])
            y.append(int(r.y))
        else:  # dict path
            f = (r.get("features") or {})
            X.append([float(f.get(k, 0.0) or 0.0) for k in feat_order])
            y.append(int(r.get("label", 0)))
    return np.array(X, dtype=float), np.array(y, dtype=int)


def _compute_coverage_from_X(X: np.ndarray, feat_order: List[str]) -> Dict[str, Dict[str, float]]:
    n = float(X.shape[0]) if X.size else 1.0
    out: Dict[str, Dict[str, float]] = {}
    if X.size == 0:
        for k in feat_order:
            out[k] = {"nonzero_pct": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return out
    for i, k in enumerate(feat_order):
        col = X[:, i]
        nz = float(np.count_nonzero(col)) / n * 100.0
        out[k] = {
            "nonzero_pct": float(nz),
            "mean": float(np.mean(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return out


def _top_coefficients_lr(model: LogisticRegression, feat_order: List[str], top: int = 5) -> List[Dict[str, float]]:
    if not hasattr(model, "coef_") or model.coef_ is None:
        return []
    coef = model.coef_.ravel().tolist()
    pairs = list(zip(feat_order, coef))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "coef": float(v)} for k, v in pairs[:top]]


def _top_importances_rf(model: RandomForestClassifier, feat_order: List[str], top: int = 5) -> List[Dict[str, float]]:
    try:
        imps = model.feature_importances_.tolist()
    except Exception:
        return []
    pairs = list(zip(feat_order, imps))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "importance": float(v)} for k, v in pairs[:top]]


def _git_sha() -> str | None:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _safe_metric(fn, y_true, y_pred, default: float = 0.0) -> float:
    try:
        return float(fn(y_true, y_pred))
    except Exception:
        return default


def train(days: int = 14, interval: str = "hour", out_dir: Path | None = None) -> Dict[str, Any]:
    """
    Trains logistic + random forest (when data exists or DEMO_MODE is on).

    Persists:
      - models/trigger_likelihood_v0.joblib + .meta.json (+ feature_coverage.json once)
      - models/trigger_likelihood_rf.joblib + .meta.json

    Returns a small dict with paths/metrics flags.
    """
    # Use current (possibly monkeypatched) paths
    out_dir = out_dir or paths.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: build log paths dynamically from LOGS_DIR so tests can monkeypatch
    log_path = paths.LOGS_DIR / "retraining_log.jsonl"
    trig_path = paths.LOGS_DIR / "retraining_triggered.jsonl"

    rows, feat_order = build_examples(
        log_path,
        trig_path,
        days=days,
        interval=interval,
    )

    demo_used = False
    if not rows and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        rows, feat_order, _ = synth_demo_dataset()
        demo_used = True

    if not rows:
        raise RuntimeError("No training rows and DEMO_MODE is false; cannot train.")

    # Arrays + coverage once
    X, y = _mk_arrays(rows, feat_order)
    coverage = _compute_coverage_from_X(X, feat_order)

    # Time-aware split
    n = X.shape[0]
    cut = max(1, int(0.8 * n))
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = X[cut:], y[cut:] if n > 1 else (X[:], y[:])

    # ---------------- Logistic ----------------
    lr = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
    )
    lr.fit(Xtr, ytr)

    p_tr_lr = lr.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va_lr = lr.predict_proba(Xva)[:, 1] if Xva.size else np.array([])

    metrics_lr = {
        "roc_auc_tr": _safe_metric(roc_auc_score, ytr, p_tr_lr, 0.5),
        "roc_auc_va": _safe_metric(roc_auc_score, yva, p_va_lr, 0.5),
        "pr_auc_tr": _safe_metric(average_precision_score, ytr, p_tr_lr, 0.0),
        "pr_auc_va": _safe_metric(average_precision_score, yva, p_va_lr, 0.0),
        "logloss_tr": _safe_metric(log_loss, ytr, p_tr_lr, 0.0),
        "logloss_va": _safe_metric(log_loss, yva, p_va_lr, 0.0),
        "brier_tr": _safe_metric(brier_score_loss, ytr, p_tr_lr, 0.0),
        "brier_va": _safe_metric(brier_score_loss, yva, p_va_lr, 0.0),
    }

    lr_model_path = out_dir / "trigger_likelihood_v0.joblib"
    lr_meta_path = out_dir / "trigger_likelihood_v0.meta.json"
    cov_path = out_dir / "feature_coverage.json"

    joblib.dump(lr, lr_model_path)
    with cov_path.open("w") as f:
        json.dump(coverage, f, indent=2)

    lr_meta: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": metrics_lr,
        "demo": bool(demo_used),
        "artifacts": {
            "model": str(lr_model_path),
            "feature_coverage": str(cov_path),
        },
        "top_features": _top_coefficients_lr(lr, feat_order, top=5),
        "feature_coverage_summary": {
            k: round(v.get("nonzero_pct", 0.0), 2) for k, v in coverage.items()
        },
        "model_type": "logistic",
    }
    with lr_meta_path.open("w") as f:
        json.dump(lr_meta, f, indent=2)

    # ---------------- Random Forest ----------------
    # Only train RF if at least 2 classes are present (avoid sklearn error)
    rf_trained = False
    rf_metrics: Dict[str, Any] = {}
    rf_model_path = out_dir / "trigger_likelihood_rf.joblib"
    rf_meta_path = out_dir / "trigger_likelihood_rf.meta.json"

    if len(set(y.tolist())) >= 2:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(Xtr, ytr)
        rf_trained = True

        p_tr_rf = rf.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
        p_va_rf = rf.predict_proba(Xva)[:, 1] if Xva.size else np.array([])

        rf_metrics = {
            "roc_auc_tr": _safe_metric(roc_auc_score, ytr, p_tr_rf, 0.5),
            "roc_auc_va": _safe_metric(roc_auc_score, yva, p_va_rf, 0.5),
            "pr_auc_tr": _safe_metric(average_precision_score, ytr, p_tr_rf, 0.0),
            "pr_auc_va": _safe_metric(average_precision_score, yva, p_va_rf, 0.0),
            "logloss_tr": _safe_metric(log_loss, ytr, p_tr_rf, 0.0),
            "logloss_va": _safe_metric(log_loss, yva, p_va_rf, 0.0),
            "brier_tr": _safe_metric(brier_score_loss, ytr, p_tr_rf, 0.0),
            "brier_va": _safe_metric(brier_score_loss, yva, p_va_rf, 0.0),
        }

        joblib.dump(rf, rf_model_path)
        rf_meta: Dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "feature_order": feat_order,
            "metrics": rf_metrics,
            "demo": bool(demo_used),
            "artifacts": {"model": str(rf_model_path)},
            "top_features": _top_importances_rf(rf, feat_order, top=5),
            "model_type": "random_forest",
        }
        with rf_meta_path.open("w") as f:
            json.dump(rf_meta, f, indent=2)

    return {
        "logistic": {
            "model_path": str(lr_model_path),
            "meta_path": str(lr_meta_path),
            "coverage_path": str(cov_path),
            "metrics": metrics_lr,
            "demo": demo_used,
        },
        "random_forest": {
            "trained": rf_trained,
            "model_path": str(rf_model_path) if rf_trained else None,
            "meta_path": str(rf_meta_path) if rf_trained else None,
            "metrics": rf_metrics if rf_trained else {},
        },
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--interval", type=str, default="hour")
    p.add_argument("--out_dir", type=str, default="")
    args = p.parse_args()

    od = Path(args.out_dir) if args.out_dir else None
    out = train(days=args.days, interval=args.interval, out_dir=od)
    print(json.dumps(out, indent=2))