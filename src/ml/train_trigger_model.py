from __future__ import annotations

import json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

from src import paths
from src.ml.feature_builder import build_examples, synth_demo_dataset


def _mk_arrays(rows: List[Any], feat_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    for r in rows:
        if hasattr(r, "x") and hasattr(r, "y"):
            X.append([float(v) for v in r.x])
            y.append(int(r.y))
        else:
            f = (r.get("features") or {})
            X.append([float(f.get(k, 0.0) or 0.0) for k in feat_order])
            y.append(int(r.get("label", 0)))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


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
            "nonzero_pct": nonzero,
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


def _git_sha() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _ensure_two_classes_in_train(
    Xtr: np.ndarray, ytr: np.ndarray, X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if Xtr.size == 0 or np.unique(ytr).size >= 2:
        return Xtr, ytr
    present = int(ytr[0])
    other = 1 - present
    idxs = np.where(y == other)[0]
    if idxs.size:
        take = idxs[: min(10, idxs.size)]
        Xtr_fix = np.vstack([Xtr, X[take]])
        ytr_fix = np.concatenate([ytr, y[take]])
        return Xtr_fix, ytr_fix
    yfix = ytr.copy()
    yfix[0] = 1 - yfix[0]
    return Xtr, yfix


def _metrics_dict(ytr, p_tr, yva, p_va) -> Dict[str, float]:
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


def _calibrate(model, Xva, yva) -> Tuple[Any, Dict[str, float]]:
    try:
        cal = CalibratedClassifierCV(base_estimator=model, method="isotonic", cv="prefit")
        cal.fit(Xva, yva)
        p_va_raw = model.predict_proba(Xva)[:, 1]
        p_va_cal = cal.predict_proba(Xva)[:, 1]
        metrics = {
            "brier_va_pre": brier_score_loss(yva, p_va_raw),
            "brier_va_post": brier_score_loss(yva, p_va_cal),
            "logloss_va_pre": log_loss(yva, p_va_raw),
            "logloss_va_post": log_loss(yva, p_va_cal),
            "roc_auc_va_pre": roc_auc_score(yva, p_va_raw),
            "roc_auc_va_post": roc_auc_score(yva, p_va_cal),
            "pr_auc_va_pre": average_precision_score(yva, p_va_raw),
            "pr_auc_va_post": average_precision_score(yva, p_va_cal),
        }
        return cal, metrics
    except Exception:
        return model, {}

def train(days: int = 14, interval: str = "hour", out_dir: Path | None = None) -> Dict[str, Any]:
    out_dir = out_dir or paths.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, feat_order = build_examples(
        paths.LOGS_DIR / "retraining_log.jsonl",
        paths.LOGS_DIR / "retraining_triggered.jsonl",
        days=days,
        interval=interval,
    )

    demo_used = False
    if not rows and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        rows, feat_order, _ = synth_demo_dataset()
        demo_used = True

    if not rows:
        raise RuntimeError("No training rows and DEMO_MODE is false; cannot train.")

    X, y = _mk_arrays(rows, feat_order)
    coverage = _compute_coverage_from_X(X, feat_order)

    n = X.shape[0]
    cut = max(1, int(0.8 * n))
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = (X[cut:], y[cut:]) if n > 1 else (X.copy(), y.copy())

    Xtr, ytr = _ensure_two_classes_in_train(Xtr, ytr, X, y)

    # ---------- Logistic ----------
    lr = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
    )
    lr.fit(Xtr, ytr)
    p_tr_lr = lr.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va_lr = lr.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    metrics_lr = _metrics_dict(ytr, p_tr_lr, yva, p_va_lr)

    # Calibration
    cal_model, cal_metrics = _calibrate(lr, Xva, yva)
    joblib.dump(cal_model, out_dir / "trigger_likelihood_v0.joblib")
    with (out_dir / "feature_coverage.json").open("w") as f:
        json.dump(coverage, f, indent=2)

    meta_lr = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": metrics_lr,
        "calibration": cal_metrics,
        "demo": bool(demo_used),
        "artifacts": {
            "model": str(out_dir / "trigger_likelihood_v0.joblib"),
            "feature_coverage": str(out_dir / "feature_coverage.json"),
        },
        "top_features": _top_coefficients(lr, feat_order, top=5),
        "feature_coverage_summary": {
            k: round(v.get("nonzero_pct", 0.0), 2) for k, v in coverage.items()
        },
    }
    with (out_dir / "trigger_likelihood_v0.meta.json").open("w") as f:
        json.dump(meta_lr, f, indent=2)

    return {
        "model_path": str(out_dir / "trigger_likelihood_v0.joblib"),
        "meta_path": str(out_dir / "trigger_likelihood_v0.meta.json"),
        "coverage_path": str(out_dir / "feature_coverage.json"),
        "metrics": metrics_lr,
        "calibration": cal_metrics,
        "demo": demo_used,
    }
