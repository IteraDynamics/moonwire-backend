# src/ml/train_trigger_model.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass  # kept for potential external use
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

# IMPORTANT: import the module, not copied constants (monkeypatch-friendly)
from src import paths
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
    Tolerant converter:
      - dict rows with {"features": {...}, "label": 0/1} OR {"x":[...], "y":0/1}
      - object rows with attributes .x (list/array) and .y (int)
    """
    X_list: List[List[float]] = []
    y_list: List[int] = []

    for r in rows:
        # Object form: e.g., FeatureRow(ts, origin, x, y)
        if hasattr(r, "x") and hasattr(r, "y"):
            xvec = [float(v) for v in list(getattr(r, "x"))]
            X_list.append(xvec)
            y_list.append(int(getattr(r, "y")))
            continue

        # Dict form with explicit vector
        if isinstance(r, dict) and "x" in r and "y" in r:
            xvec = [float(v) for v in list(r.get("x") or [])]
            X_list.append(xvec)
            y_list.append(int(r.get("y", 0)))
            continue

        # Dict form with feature map + label
        if isinstance(r, dict):
            f = r.get("features", {}) or {}
            xvec = [float(f.get(k, 0.0) or 0.0) for k in feat_order]
            yval = r.get("label", r.get("y", 0))
            X_list.append(xvec)
            y_list.append(int(yval))
            continue

        # Fallback (shouldn’t happen): skip row
        continue

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)
    return X, y


def _compute_coverage_from_X(X: np.ndarray, feat_order: List[str]) -> Dict[str, Dict[str, float]]:
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
    # Use paths.MODELS_DIR unless an explicit out_dir was provided (monkeypatch-friendly)
    out_dir = out_dir or paths.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute log file paths at call time from the CURRENT LOGS_DIR (monkeypatch-friendly)
    flags_path = paths.LOGS_DIR / "retraining_log.jsonl"
    triggers_path = paths.LOGS_DIR / "retraining_triggered.jsonl"

    # Build examples from those paths
    rows, feat_order = build_examples(
        flags_path,
        triggers_path,
        days=days,
        interval=interval,
    )

    demo_used = False
    if not rows and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        rows, feat_order, _ = synth_demo_dataset()
        demo_used = True

    if not rows:
        raise RuntimeError("No training rows and DEMO_MODE is false; cannot train.")

    # Arrays + coverage (coverage over the full dataset used for training)
    X, y = _mk_arrays(rows, feat_order)
    coverage = _compute_coverage_from_X(X, feat_order)

    # Simple time-aware split: first 80% train, last 20% validate
    n = int(X.shape[0])
    cut = max(1, int(0.8 * n))
    if n > 1:
        Xtr, ytr = X[:cut], y[:cut]
        Xva, yva = X[cut:], y[cut:]
    else:
        Xtr, ytr = X, y
        Xva, yva = X, y

    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        n_jobs=None,
    )
    clf.fit(Xtr, ytr)

    def _safe_metric(fn, y_true, y_pred, default: float = 0.0) -> float:
        try:
            return float(fn(y_true, y_pred))
        except Exception:
            return default

    # Predict probabilities
    p_tr = clf.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va = clf.predict_proba(Xva)[:, 1] if Xva.size else np.array([])

    metrics = {
        "roc_auc_tr": _safe_metric(roc_auc_score, ytr, p_tr, 0.5),
        "roc_auc_va": _safe_metric(roc_auc_score, yva, p_va, 0.5),
        "pr_auc_tr": _safe_metric(average_precision_score, ytr, p_tr, 0.0),
        "pr_auc_va": _safe_metric(average_precision_score, yva, p_va, 0.0),
        "logloss_tr": _safe_metric(log_loss, ytr, p_tr, 0.0),
        "logloss_va": _safe_metric(log_loss, yva, p_va, 0.0),
        "brier_tr": _safe_metric(brier_score_loss, ytr, p_tr, 0.0),
        "brier_va": _safe_metric(brier_score_loss, yva, p_va, 0.0),
    }

    # Persist artifacts
    model_path = out_dir / "trigger_likelihood_v0.joblib"
    meta_path = out_dir / "trigger_likelihood_v0.meta.json"
    coverage_path = out_dir / "feature_coverage.json"

    joblib.dump(clf, model_path)

    with coverage_path.open("w") as f:
        json.dump(coverage, f, indent=2)

    meta: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": metrics,
        "demo": bool(demo_used),
        "artifacts": {
            "model": str(model_path),
            "feature_coverage": str(coverage_path),
        },
        "top_features": _top_coefficients(clf, feat_order, top=5),
        "feature_coverage_summary": {k: round(v.get("nonzero_pct", 0.0), 2) for k, v in coverage.items()},
    }

    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "coverage_path": str(coverage_path),
        "metrics": metrics,
        "demo": demo_used,
    }