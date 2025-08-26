# src/ml/train_trigger_model.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src import paths  # <- module import so monkeypatching works in tests
from src.ml.feature_builder import build_examples, synth_demo_dataset


# ----------------------------- helpers ---------------------------------
def _git_sha() -> str | None:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
            or None
        )
    except Exception:
        return None


def _mk_arrays(
    rows: List[Any], feat_order: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accept rows as either dicts like {"features": {...}, "label": 0/1}
    or simple containers with attributes x (list/array) and y (int).
    """
    X: List[List[float]] = []
    y: List[int] = []
    for r in rows:
        if isinstance(r, dict):
            f = (r.get("features") or {}) if "features" in r else r
            X.append([float(f.get(k, 0.0) or 0.0) for k in feat_order])
            y.append(int(r.get("label", 0)))
        else:
            # dataclass-like: FeatureRow(ts, origin, x, y)
            fx = getattr(r, "x", None)
            fy = getattr(r, "y", None)
            if fx is None or fy is None:
                continue
            X.append([float(v) for v in list(fx)])
            y.append(int(fy))
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=int)
    return X_arr, y_arr


def _compute_coverage_from_X(
    X: np.ndarray, feat_order: List[str]
) -> Dict[str, Dict[str, float]]:
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


def _top_coefficients_from_linear(
    model: LogisticRegression, feat_order: List[str], top: int = 5
) -> List[Dict[str, float]]:
    try:
        coef = model.coef_.ravel().tolist()
    except Exception:
        return []
    pairs = list(zip(feat_order, coef))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "coef": float(v)} for k, v in pairs[:top]]


def _top_importances(
    model, feat_order: List[str], top: int = 5
) -> List[Dict[str, float]]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return []
    pairs = list(zip(feat_order, [float(v) for v in imp]))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "importance": float(v)} for k, v in pairs[:top]]


def _ensure_two_classes(
    X: np.ndarray, y: np.ndarray, feat_order: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """If y has only a single class, append a tiny synthetic opposite-class row."""
    if y.size == 0:
        return X, y
    if np.unique(y).size >= 2:
        return X, y
    # Synthesize a single positive if all zeros (or vice versa)
    xsyn = np.zeros((1, X.shape[1]), dtype=float)
    # Try to bump 'burst_z' if present to give the model a signal
    try:
        idx = feat_order.index("burst_z")
        xsyn[0, idx] = 1.0
    except Exception:
        pass
    ysyn = 1 if int(y[0]) == 0 else 0
    X2 = np.vstack([X, xsyn])
    y2 = np.concatenate([y, np.array([ysyn], dtype=int)])
    return X2, y2


def _train_and_report_linear(
    Xtr, ytr, Xva, yva, feat_order: List[str]
) -> Tuple[LogisticRegression, Dict[str, float], List[Dict[str, float]]]:
    lr = LogisticRegression(
        solver="liblinear", class_weight="balanced", max_iter=2000
    )
    # Guard against single-class splits
    Xtr2, ytr2 = _ensure_two_classes(Xtr, ytr, feat_order)
    lr.fit(Xtr2, ytr2)
    p_tr = lr.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va = lr.predict_proba(Xva)[:, 1] if Xva.size else np.array([])

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
    top_feats = _top_coefficients_from_linear(lr, feat_order, top=5)
    return lr, metrics, top_feats


def _train_and_report_rf(
    Xtr, ytr, Xva, yva, feat_order: List[str]
) -> Tuple[RandomForestClassifier, Dict[str, float], List[Dict[str, float]]]:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    Xtr2, ytr2 = _ensure_two_classes(Xtr, ytr, feat_order)
    rf.fit(Xtr2, ytr2)
    # For RF, get calibrated probabilities via predict_proba
    p_tr = rf.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va = rf.predict_proba(Xva)[:, 1] if Xva.size else np.array([])

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
    top_feats = _top_importances(rf, feat_order, top=5)
    return rf, metrics, top_feats


def _train_and_report_gb(
    Xtr, ytr, Xva, yva, feat_order: List[str]
) -> Tuple[GradientBoostingClassifier, Dict[str, float], List[Dict[str, float]]]:
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    )
    Xtr2, ytr2 = _ensure_two_classes(Xtr, ytr, feat_order)
    gb.fit(Xtr2, ytr2)
    # GradientBoostingClassifier has predict_proba
    p_tr = gb.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va = gb.predict_proba(Xva)[:, 1] if Xva.size else np.array([])

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
    top_feats = _top_importances(gb, feat_order, top=5)
    return gb, metrics, top_feats


def _write_meta(
    meta_path: Path,
    model_path: Path,
    feat_order: List[str],
    metrics: Dict[str, float],
    coverage: Dict[str, Dict[str, float]] | None,
    top_features: List[Dict[str, float]],
    demo_used: bool,
) -> None:
    meta: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": metrics,
        "demo": bool(demo_used),
        "artifacts": {
            "model": str(model_path),
        },
        "top_features": top_features,
    }
    if coverage is not None:
        meta["artifacts"]["feature_coverage"] = str(
            (meta_path.parent / "feature_coverage.json")
        )
        meta["feature_coverage_summary"] = {
            k: round(v.get("nonzero_pct", 0.0), 2) for k, v in coverage.items()
        }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


# ----------------------------- main train() ---------------------------------
def train(days: int = 14, interval: str = "hour", out_dir: Path | None = None) -> Dict[str, Any]:
    """
    Train logistic + random forest + gradient boosting (when data exists or DEMO_MODE is on).
    Persists artifacts for each model; coverage json is written once.
    Returns a dict containing at least the logistic paths/metrics for back-compat with tests.
    """
    out_dir = out_dir or paths.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build examples using live paths (monkeypatch-friendly)
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
    coverage = _compute_coverage_from_X(X, feat_order)
    cov_path = out_dir / "feature_coverage.json"
    with cov_path.open("w") as f:
        json.dump(coverage, f, indent=2)

    # Simple time-aware split
    n = X.shape[0]
    cut = max(1, int(0.8 * n))
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = (X[cut:], y[cut:]) if n > 1 else (X[:], y[:])

    # ---- Logistic
    lr, lr_metrics, lr_top = _train_and_report_linear(Xtr, ytr, Xva, yva, feat_order)
    lr_model_path = out_dir / "trigger_likelihood_v0.joblib"
    lr_meta_path = out_dir / "trigger_likelihood_v0.meta.json"
    joblib.dump(lr, lr_model_path)
    _write_meta(
        lr_meta_path,
        lr_model_path,
        feat_order,
        lr_metrics,
        coverage,
        lr_top,
        demo_used,
    )

    # ---- Random Forest (best-effort; continue if it fails)
    rf_model_path = out_dir / "trigger_likelihood_rf.joblib"
    rf_meta_path = out_dir / "trigger_likelihood_rf.meta.json"
    try:
        rf, rf_metrics, rf_top = _train_and_report_rf(Xtr, ytr, Xva, yva, feat_order)
        joblib.dump(rf, rf_model_path)
        _write_meta(
            rf_meta_path,
            rf_model_path,
            feat_order,
            rf_metrics,
            None,  # coverage already saved once; we still include summary in meta
            rf_top,
            demo_used,
        )
    except Exception:
        # Skip silently; logistic is our baseline
        rf_model_path = None
        rf_meta_path = None

    # ---- Gradient Boosting (best-effort; continue if it fails)
    gb_model_path = out_dir / "trigger_likelihood_gb.joblib"
    gb_meta_path = out_dir / "trigger_likelihood_gb.meta.json"
    try:
        gb, gb_metrics, gb_top = _train_and_report_gb(Xtr, ytr, Xva, yva, feat_order)
        joblib.dump(gb, gb_model_path)
        _write_meta(
            gb_meta_path,
            gb_model_path,
            feat_order,
            gb_metrics,
            None,
            gb_top,
            demo_used,
        )
    except Exception:
        gb_model_path = None
        gb_meta_path = None

    # Return (keep keys used by existing tests)
    out: Dict[str, Any] = {
        "model_path": str(lr_model_path),
        "meta_path": str(lr_meta_path),
        "coverage_path": str(cov_path),
        "metrics": lr_metrics,
        "demo": demo_used,
    }
    if rf_model_path:
        out["rf_model_path"] = str(rf_model_path)
        out["rf_meta_path"] = str(rf_meta_path)
    if gb_model_path:
        out["gb_model_path"] = str(gb_model_path)
        out["gb_meta_path"] = str(gb_meta_path)
    return out


# ----------------------------- CLI entrypoint ---------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train trigger likelihood models")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--interval", type=str, default="hour", choices=["hour", "day"])
    p.add_argument("--out_dir", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else None
    res = train(days=args.days, interval=args.interval, out_dir=out_dir)
    print(json.dumps(res, indent=2))