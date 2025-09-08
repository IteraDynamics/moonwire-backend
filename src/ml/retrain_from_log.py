# src/ml/retrain_from_log.py
from __future__ import annotations

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
    log_loss,
    roc_auc_score,
    brier_score_loss,
)

# --------------------
# Config knobs (env)
# --------------------
DEFAULT_TRAIN_LOG = os.getenv("TRAIN_LOG_PATH", "models/training_data.jsonl")
DEFAULT_MODELS_DIR = os.getenv("SAVE_MODEL_DIR", "models")
DEFAULT_MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.5.0")
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "50"))  # low-sample guard (non-blocking)


# --------------------
# Small utils
# --------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_metric(fn, y_true, y_prob, default: float = 0.0) -> float:
    try:
        return float(fn(y_true, y_prob))
    except Exception:
        return default


def _mk_arrays(rows: List[Dict[str, Any]], feat_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    for r in rows:
        feats = r.get("features") or {}
        X.append([float(feats.get(k, 0.0) or 0.0) for k in feat_order])
        y.append(1 if bool(r.get("label")) else 0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def _ensure_two_classes_in_train(
    Xtr: np.ndarray, ytr: np.ndarray, X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """If the training split has only one class, borrow some examples from the other split or flip one label."""
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


def _git_sha() -> str | None:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


@dataclass
class SavePaths:
    base_dir: Path
    version_dir: Path
    logistic_path: Path
    logistic_meta: Path
    rf_path: Path
    rf_meta: Path
    gb_path: Path
    gb_meta: Path
    coverage_path: Path


def _prepare_out_dirs(models_dir: Path, version: str) -> SavePaths:
    base = models_dir
    version_dir = base / version.replace(".", "")
    version_dir.mkdir(parents=True, exist_ok=True)

    return SavePaths(
        base_dir=base,
        version_dir=version_dir,
        logistic_path=version_dir / "trigger_likelihood_v0.joblib",
        logistic_meta=version_dir / "trigger_likelihood_v0.meta.json",
        rf_path=version_dir / "trigger_likelihood_rf.joblib",
        rf_meta=version_dir / "trigger_likelihood_rf.meta.json",
        gb_path=version_dir / "trigger_likelihood_gb.joblib",
        gb_meta=version_dir / "trigger_likelihood_gb.meta.json",
        coverage_path=version_dir / "feature_coverage.json",
    )


# --------------------
# Main entry point
# --------------------
def retrain_from_log(
    train_log_path: str | Path = DEFAULT_TRAIN_LOG,
    out_dir: str | Path = DEFAULT_MODELS_DIR,
    version: str = DEFAULT_MODEL_VERSION,
) -> Dict[str, Any]:
    train_log_path = Path(train_log_path)
    out_dir = Path(out_dir)

    # Load rows
    if not train_log_path.exists():
        raise RuntimeError(f"{train_log_path} does not exist; cannot retrain.")
    rows = [json.loads(x) for x in train_log_path.read_text().splitlines() if x.strip()]
    n_rows = len(rows)
    if n_rows == 0:
        raise RuntimeError(f"No rows in {train_log_path}; cannot retrain.")

    # Low-sample (non-blocking) flag
    low_sample = n_rows < MIN_TRAIN_ROWS

    # Infer feature order from the first row
    first_feats = (rows[0].get("features") or {})
    feat_order = list(first_feats.keys())

    # Build arrays
    X, y = _mk_arrays(rows, feat_order)

    # Split
    n = X.shape[0]
    cut = max(1, int(0.8 * n))
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = (X[cut:], y[cut:]) if n > 1 else (X.copy(), y.copy())

    # Ensure two classes in train
    Xtr, ytr = _ensure_two_classes_in_train(Xtr, ytr, X, y)

    # Simple coverage summary from X
    coverage_summary = {}
    if X.size:
        nrows = float(X.shape[0])
        for i, k in enumerate(feat_order):
            col = X[:, i]
            nz_pct = float(np.count_nonzero(col)) / max(1.0, nrows) * 100.0
            coverage_summary[k] = round(nz_pct, 2)

    # Prepare output paths
    paths = _prepare_out_dirs(out_dir, version)

    # ----------------
    # Train Logistic
    # ----------------
    lr = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
    )
    lr.fit(Xtr, ytr)
    p_tr_lr = lr.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va_lr = lr.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    metrics_lr = _metrics_dict(ytr, p_tr_lr, yva, p_va_lr)
    joblib.dump(lr, paths.logistic_path)
    paths.coverage_path.write_text(json.dumps({k: {"nonzero_pct": v} for k, v in coverage_summary.items()}, indent=2))

    meta_lr: Dict[str, Any] = {
        "created_at": _now_iso(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": metrics_lr,
        "demo": False,
        "artifacts": {
            "model": str(paths.logistic_path.resolve()),
            "feature_coverage": str(paths.coverage_path.resolve()),
        },
        "top_features": _top_coefficients(lr, feat_order, top=5),
        "feature_coverage_summary": coverage_summary,
        "training_rows": n_rows,
        "low_sample": bool(low_sample),
        "version": version,
    }
    paths.logistic_meta.write_text(json.dumps(meta_lr, indent=2))

    # ----------------
    # Train Random Forest
    # ----------------
    rf_trained = False
    metrics_rf: Dict[str, float] | None = None
    try:
        if np.unique(y).size >= 2:
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(Xtr, ytr)
            p_tr_rf = rf.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
            p_va_rf = rf.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
            metrics_rf = _metrics_dict(ytr, p_tr_rf, yva, p_va_rf)
            joblib.dump(rf, paths.rf_path)
            meta_rf = {
                "created_at": _now_iso(),
                "git_sha": _git_sha(),
                "feature_order": feat_order,
                "metrics": metrics_rf,
                "demo": False,
                "artifacts": {
                    "model": str(paths.rf_path.resolve()),
                    "feature_coverage": str(paths.coverage_path.resolve()),
                },
                "training_rows": n_rows,
                "low_sample": bool(low_sample),
                "version": version,
            }
            paths.rf_meta.write_text(json.dumps(meta_rf, indent=2))
            rf_trained = True
    except Exception:
        rf_trained = False
        metrics_rf = None

    # ----------------
    # Train Gradient Boosting
    # ----------------
    gb_trained = False
    metrics_gb: Dict[str, float] | None = None
    try:
        if np.unique(y).size >= 2:
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
            joblib.dump(gb, paths.gb_path)
            meta_gb = {
                "created_at": _now_iso(),
                "git_sha": _git_sha(),
                "feature_order": feat_order,
                "metrics": metrics_gb,
                "demo": False,
                "artifacts": {
                    "model": str(paths.gb_path.resolve()),
                    "feature_coverage": str(paths.coverage_path.resolve()),
                },
                "training_rows": n_rows,
                "low_sample": bool(low_sample),
                "version": version,
            }
            paths.gb_meta.write_text(json.dumps(meta_gb, indent=2))
            gb_trained = True
    except Exception:
        gb_trained = False
        metrics_gb = None

    # ----------------
    # Optional: write/refresh top-level (non-versioned) symlinks
    # (safe no-op on systems that don't allow symlinks)
    # ----------------
    try:
        def _symlink(src: Path, link: Path):
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src)

        _symlink(paths.logistic_path, out_dir / "trigger_likelihood_v0.joblib")
        _symlink(paths.logistic_meta, out_dir / "trigger_likelihood_v0.meta.json")
        if rf_trained:
            _symlink(paths.rf_path, out_dir / "trigger_likelihood_rf.joblib")
            _symlink(paths.rf_meta, out_dir / "trigger_likelihood_rf.meta.json")
        if gb_trained:
            _symlink(paths.gb_path, out_dir / "trigger_likelihood_gb.joblib")
            _symlink(paths.gb_meta, out_dir / "trigger_likelihood_gb.meta.json")
        _symlink(paths.coverage_path, out_dir / "feature_coverage.json")
    except Exception:
        pass

    return {
        "version": version,
        "training_rows": n_rows,
        "low_sample": bool(low_sample),
        "artifacts": {
            "dir": str(paths.version_dir),
            "logistic": str(paths.logistic_path),
            "rf": str(paths.rf_path) if rf_trained else None,
            "gb": str(paths.gb_path) if gb_trained else None,
        },
        "metrics": {
            "logistic": metrics_lr,
            "rf": metrics_rf,
            "gb": metrics_gb,
        },
        "feature_order": feat_order,
        "created_at": meta_lr["created_at"],
    }


# --------------------
# CLI
# --------------------
if __name__ == "__main__":
    out = retrain_from_log(
        train_log_path=DEFAULT_TRAIN_LOG,
        out_dir=DEFAULT_MODELS_DIR,
        version=DEFAULT_MODEL_VERSION,
    )
    # Minimal console summary (use CI summary for rich display)
    print(
        json.dumps(
            {
                "version": out["version"],
                "training_rows": out["training_rows"],
                "low_sample": out["low_sample"],
                "created_at": out["created_at"],
                "models": [k for k, v in out["metrics"].items() if v],
            },
            indent=2,
        )
    )