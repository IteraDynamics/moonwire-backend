from __future__ import annotations

import json, os
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

from sklearn.calibration import CalibratedClassifierCV  # NEW

from src import paths
from src.ml.feature_builder import build_examples, synth_demo_dataset


def _mk_arrays(rows: List[Any], feat_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    for r in rows:
        if hasattr(r, "x") and hasattr(r, "y"):  # dataclass route
            X.append([float(v) for v in r.x])
            y.append(int(r.y))
        else:  # dict route
            f = (r.get("features") or {})
            X.append([float(f.get(k, 0.0) or 0.0) for k in feat_order])
            y.append(int(r.get("label", 0)))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def _compute_coverage_from_X(X: np.ndarray, feat_order: List[str]) -> Dict[str, Dict[str, float]]:
    n = float(X.shape[0]) if X.size else 1.0
    out: Dict[str, Dict[str, float]] = {}
    if X.size == 0:
        for k in feat_order:
            out[k] = {"nonzero_pct": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return out
    for i, k in enumerate(feat_order):
        col = X[:, i]
        nonzero = float(np.count_nonzero(col)) / n * 100.0
        out[k] = {
            "nonzero_pct": nonzero,
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
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


def _calibrate_on_validation(
    base_model,
    Xva: np.ndarray,
    yva: np.ndarray,
    method: str = "isotonic",
) -> Tuple[Dict[str, float], object | None]:
    """
    Calibrate a prefit classifier on the validation set using CalibratedClassifierCV.
    Returns (calibration_metrics_dict, calibrated_model_or_None).
    The metrics are computed on the same validation set for a simple CI-friendly report.
    """
    calib_report: Dict[str, float | str] = {
        "method": method,
        "on": "validation",
    }

    try:
        if Xva.size == 0 or np.unique(yva).size < 2:
            calib_report["note"] = "skipped: validation lacks two classes or is empty"
            return calib_report, None

        # Pre-calibration predictions (validation)
        p_pre = base_model.predict_proba(Xva)[:, 1]

        # Fit calibration model using the prefit base model
        method = method.lower().strip()
        if method not in ("isotonic", "sigmoid"):
            method = "isotonic"
        calibrator = CalibratedClassifierCV(base_model, method=method, cv="prefit")
        calibrator.fit(Xva, yva)

        p_post = calibrator.predict_proba(Xva)[:, 1]

        # Pre metrics
        calib_report["roc_auc_pre"] = _safe_metric(roc_auc_score, yva, p_pre, 0.5)
        calib_report["pr_auc_pre"] = _safe_metric(average_precision_score, yva, p_pre, 0.0)
        calib_report["logloss_pre"] = _safe_metric(log_loss, yva, p_pre, 0.0)
        calib_report["brier_pre"] = _safe_metric(brier_score_loss, yva, p_pre, 0.0)

        # Post metrics
        calib_report["roc_auc_post"] = _safe_metric(roc_auc_score, yva, p_post, 0.5)
        calib_report["pr_auc_post"] = _safe_metric(average_precision_score, yva, p_post, 0.0)
        calib_report["logloss_post"] = _safe_metric(log_loss, yva, p_post, 0.0)
        calib_report["brier_post"] = _safe_metric(brier_score_loss, yva, p_post, 0.0)

        return calib_report, calibrator

    except Exception as e:
        calib_report["note"] = f"skipped: {type(e).__name__}: {e}"
        return calib_report, None


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

    # Arrays + coverage
    X, y = _mk_arrays(rows, feat_order)
    coverage = _compute_coverage_from_X(X, feat_order)

    # Time-aware split
    n = X.shape[0]
    cut = max(1, int(0.8 * n))
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = (X[cut:], y[cut:]) if n > 1 else (X.copy(), y.copy())

    # Ensure 2 classes in train
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

    # --- Calibration pass (validation-based) ---
    calib_method = os.getenv("CALIBRATION_METHOD", "isotonic")
    calib_info, lr_calibrated = _calibrate_on_validation(lr, Xva, yva, method=calib_method)

    # Persist models & meta
    model_path = out_dir / "trigger_likelihood_v0.joblib"
    meta_path = out_dir / "trigger_likelihood_v0.meta.json"
    coverage_path = out_dir / "feature_coverage.json"
    joblib.dump(lr, model_path)

    # Save calibrated model if available (optional, for future inference)
    calibrated_model_path: str | None = None
    if lr_calibrated is not None:
        calibrated_model_path = str(out_dir / "trigger_likelihood_v0.calibrated.joblib")
        try:
            joblib.dump(lr_calibrated, calibrated_model_path)
        except Exception:
            calibrated_model_path = None

    with coverage_path.open("w") as f:
        json.dump(coverage, f, indent=2)

    meta_lr: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": metrics_lr,
        "calibration": calib_info or {},                     # <-- filled now
        "demo": bool(demo_used),
        "artifacts": {
            "model": str(model_path),
            "calibrated_model": calibrated_model_path,       # may be None
            "feature_coverage": str(coverage_path),
        },
        "top_features": _top_coefficients(lr, feat_order, top=5),
        "feature_coverage_summary": {
            k: round(v.get("nonzero_pct", 0.0), 2) for k, v in coverage.items()
        },
    }
    with meta_path.open("w") as f:
        json.dump(meta_lr, f, indent=2)

    # ---------- Random Forest (optional) ----------
    rf_trained = False
    rf_model_path = out_dir / "trigger_likelihood_rf.joblib"
    rf_meta_path = out_dir / "trigger_likelihood_rf.meta.json"
    metrics_rf: Dict[str, float] | None = None
    try:
        if np.unique(y).size >= 2:
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(Xtr, ytr)
            p_tr_rf = rf.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
            p_va_rf = rf.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
            metrics_rf = _metrics_dict(ytr, p_tr_rf, yva, p_va_rf)
            joblib.dump(rf, rf_model_path)

            meta_rf = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": _git_sha(),
                "feature_order": feat_order,
                "metrics": metrics_rf,
                "demo": bool(demo_used),
                "artifacts": {
                    "model": str(rf_model_path),
                    "feature_coverage": str(coverage_path),
                },
            }
            with rf_meta_path.open("w") as f:
                json.dump(meta_rf, f, indent=2)
            rf_trained = True
    except Exception:
        rf_trained = False
        metrics_rf = None

    # ---------- Gradient Boosting (optional) ----------
    gb_trained = False
    gb_model_path = out_dir / "trigger_likelihood_gb.joblib"
    gb_meta_path = out_dir / "trigger_likelihood_gb.meta.json"
    metrics_gb: Dict[str, float] | None = None
    try:
        if np.unique(y).size >= 2:
            from sklearn.ensemble import GradientBoostingClassifier

            gb = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
            gb.fit(Xtr, ytr)
            p_tr_gb = gb.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
            p_va_gb = gb.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
            metrics_gb = _metrics_dict(ytr, p_tr_gb, yva, p_va_gb)
            joblib.dump(gb, gb_model_path)

            meta_gb = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": _git_sha(),
                "feature_order": feat_order,
                "metrics": metrics_gb,
                "demo": bool(demo_used),
                "artifacts": {
                    "model": str(gb_model_path),
                    "feature_coverage": str(coverage_path),
                },
                "top_features": [],  # GB doesn't provide native coefficients
            }
            with gb_meta_path.open("w") as f:
                json.dump(meta_gb, f, indent=2)
            gb_trained = True
    except Exception:
        gb_trained = False
        metrics_gb = None

    # ---------- Return ----------
    result: Dict[str, Any] = {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "coverage_path": str(coverage_path),
        "metrics": metrics_lr,
        "demo": demo_used,
    }
    result["rf_model_path"] = str(rf_model_path) if rf_trained else None
    result["rf_meta_path"] = str(rf_meta_path) if rf_trained else None
    result["rf_metrics"] = metrics_rf
    result["gb_model_path"] = str(gb_model_path) if gb_trained else None
    result["gb_meta_path"] = str(gb_meta_path) if gb_trained else None
    result["gb_metrics"] = metrics_gb
    # for convenience in debugging
    result["calibration"] = calib_info
    result["calibrated_model_path"] = calibrated_model_path
    return result
    
# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse, json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Train Trigger Likelihood models")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--interval", type=str, default="hour")
    ap.add_argument("--out-dir", type=str, default=None, help="Optional output dir (defaults to paths.MODELS_DIR)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    result = train(days=args.days, interval=args.interval, out_dir=out_dir)

    # Print a compact summary for CI logs
    print(json.dumps({
        "model_path": result.get("model_path"),
        "rf_model_path": result.get("rf_model_path"),
        "gb_model_path": result.get("gb_model_path"),
        "demo": result.get("demo"),
    }, indent=2))