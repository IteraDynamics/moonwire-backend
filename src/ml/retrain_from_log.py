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
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.paths import MODELS_DIR

# ----------- config / paths -----------
DEFAULT_TRAIN_LOG = MODELS_DIR / "training_data.jsonl"


# ----------- helpers -----------
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str | None:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return None


def _safe_metric(fn, y_true, y_pred, default: float = 0.0) -> float:
    try:
        return float(fn(y_true, y_pred))
    except Exception:
        return default


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


def _top_coefficients(
    model: LogisticRegression, feat_order: List[str], top: int = 5
) -> List[Dict[str, float]]:
    if not hasattr(model, "coef_") or model.coef_ is None:
        return []
    coef = model.coef_.ravel().tolist()
    pairs = list(zip(feat_order, coef))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "coef": float(v)} for k, v in pairs[:top]]


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out: List[dict] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def _build_Xy(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Expect rows like:
      {"timestamp": "...", "origin": "...", "features": {...}, "label": true/false}
    """
    # Resolve feature order (stable): union of keys across rows, sorted
    all_keys = set()
    for r in rows:
        feats = r.get("features") or {}
        all_keys.update(k for k in feats.keys())
    feat_order = sorted(all_keys)

    X_list: List[List[float]] = []
    y_list: List[int] = []
    for r in rows:
        f = r.get("features") or {}
        X_list.append([float(f.get(k, 0.0) or 0.0) for k in feat_order])
        y_list.append(1 if bool(r.get("label")) else 0)

    X = np.asarray(X_list, dtype=float) if X_list else np.zeros((0, len(feat_order)))
    y = np.asarray(y_list, dtype=int) if y_list else np.zeros((0,), dtype=int)
    return X, y, feat_order


def _ensure_two_classes_in_train(
    Xtr: np.ndarray, ytr: np.ndarray, X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If train split is single-class, borrow a few examples from the other class in the
    full set to stabilize training. If none exist, flip one label as a last resort.
    """
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
    # last resort
    yfix = ytr.copy()
    yfix[0] = 1 - yfix[0]
    return Xtr, yfix


@dataclass
class TrainOutput:
    status: str
    rows: int
    version_dir: str | None
    logistic_meta: Dict[str, Any] | None
    rf_meta: Dict[str, Any] | None
    gb_meta: Dict[str, Any] | None


# ----------- main retrain function -----------
def retrain_from_log(
    train_log_path: Path | None = None,
    out_root: Path | None = None,
    model_version: str | None = None,
) -> Dict[str, Any]:
    """
    Retrain from models/training_data.jsonl (or provided path).
    Produces: logistic (required), rf, gb (best-effort) with metadata.
    """
    train_log_path = train_log_path or Path(
        os.getenv("TRAIN_LOG_PATH", str(DEFAULT_TRAIN_LOG))
    )
    out_root = out_root or MODELS_DIR
    model_version = model_version or os.getenv("MODEL_VERSION", "v0.5.0")

    rows = _read_jsonl(train_log_path)
    n_rows = len(rows)

    # ---- low-sample guard (CI-safe) ----
    min_rows = int(os.getenv("MODEL_RETRAIN_MIN_ROWS", "10"))
    if n_rows < min_rows:
        return {
            "status": "skipped_low_sample",
            "rows": n_rows,
            "min_rows": min_rows,
            "message": f"Not enough rows ({n_rows}<{min_rows}); retrain skipped.",
        }

    # Build matrix
    X, y, feat_order = _build_Xy(rows)
    if X.size == 0 or y.size == 0:
        raise RuntimeError("Empty X/y after parsing training data; cannot retrain.")

    # Simple 80/20 split
    n = X.shape[0]
    cut = max(1, int(0.8 * n))
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = (X[cut:], y[cut:]) if n > 1 else (X.copy(), y.copy())
    Xtr, ytr = _ensure_two_classes_in_train(Xtr, ytr, X, y)

    # Versioned output dir
    version_dir = out_root / model_version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Coverage (simple)
    coverage: Dict[str, Dict[str, float]] = {}
    nrows = float(X.shape[0]) if X.size else 1.0
    for i, k in enumerate(feat_order):
        col = X[:, i]
        nonzero = float(np.count_nonzero(col)) / nrows * 100.0
        coverage[k] = {
            "nonzero_pct": nonzero,
            "mean": float(np.mean(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }

    # ---------- Logistic (required) ----------
    lr_model_path = version_dir / "trigger_likelihood_v0.joblib"
    lr_meta_path = version_dir / "trigger_likelihood_v0.meta.json"
    cov_path = version_dir / "feature_coverage.json"

    lr = LogisticRegression(
        solver="liblinear", class_weight="balanced", max_iter=2000
    )
    lr.fit(Xtr, ytr)
    p_tr_lr = lr.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va_lr = lr.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    metrics_lr = _metrics_dict(ytr, p_tr_lr, yva, p_va_lr)

    joblib.dump(lr, lr_model_path)
    cov_path.write_text(json.dumps(coverage, indent=2))

    meta_lr: Dict[str, Any] = {
        "created_at": _utcnow_iso(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": metrics_lr,
        "demo": False,
        "artifacts": {
            "model": str(lr_model_path),
            "feature_coverage": str(cov_path),
        },
        "top_features": _top_coefficients(lr, feat_order, top=5),
        "feature_coverage_summary": {
            k: round(v.get("nonzero_pct", 0.0), 2) for k, v in coverage.items()
        },
        # keep placeholder for calibration block (not part of this function)
        "calibration": {},
    }
    lr_meta_path.write_text(json.dumps(meta_lr, indent=2))

    # ---------- Random Forest (best-effort) ----------
    rf_meta: Dict[str, Any] | None = None
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

            rf_model_path = version_dir / "trigger_likelihood_rf.joblib"
            rf_meta_path = version_dir / "trigger_likelihood_rf.meta.json"
            joblib.dump(rf, rf_model_path)

            rf_meta = {
                "created_at": _utcnow_iso(),
                "git_sha": _git_sha(),
                "feature_order": feat_order,
                "metrics": metrics_rf,
                "demo": False,
                "artifacts": {
                    "model": str(rf_model_path),
                    "feature_coverage": str(cov_path),
                },
            }
            rf_meta_path.write_text(json.dumps(rf_meta, indent=2))
    except Exception:
        rf_meta = None

    # ---------- Gradient Boosting (best-effort) ----------
    gb_meta: Dict[str, Any] | None = None
    try:
        if np.unique(y).size >= 2:
            gb = GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
            )
            gb.fit(Xtr, ytr)
            p_tr_gb = gb.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
            p_va_gb = gb.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
            metrics_gb = _metrics_dict(ytr, p_tr_gb, yva, p_va_gb)

            gb_model_path = version_dir / "trigger_likelihood_gb.joblib"
            gb_meta_path = version_dir / "trigger_likelihood_gb.meta.json"
            joblib.dump(gb, gb_model_path)

            gb_meta = {
                "created_at": _utcnow_iso(),
                "git_sha": _git_sha(),
                "feature_order": feat_order,
                "metrics": metrics_gb,
                "demo": False,
                "artifacts": {
                    "model": str(gb_model_path),
                    "feature_coverage": str(cov_path),
                },
                "top_features": [],  # no direct importances here
            }
            gb_meta_path.write_text(json.dumps(gb_meta, indent=2))
    except Exception:
        gb_meta = None

    # ---------- also update unversioned "latest" copies for back-compat ----------
    # (safe best-effort; ignore errors)
    try:
        (MODELS_DIR / "trigger_likelihood_v0.joblib").write_bytes(
            lr_model_path.read_bytes()
        )
        (MODELS_DIR / "trigger_likelihood_v0.meta.json").write_text(
            lr_meta_path.read_text()
        )
        (MODELS_DIR / "feature_coverage.json").write_text(cov_path.read_text())

        if rf_meta:
            (MODELS_DIR / "trigger_likelihood_rf.joblib").write_bytes(
                (version_dir / "trigger_likelihood_rf.joblib").read_bytes()
            )
            (MODELS_DIR / "trigger_likelihood_rf.meta.json").write_text(
                (version_dir / "trigger_likelihood_rf.meta.json").read_text()
            )
        if gb_meta:
            (MODELS_DIR / "trigger_likelihood_gb.joblib").write_bytes(
                (version_dir / "trigger_likelihood_gb.joblib").read_bytes()
            )
            (MODELS_DIR / "trigger_likelihood_gb.meta.json").write_text(
                (version_dir / "trigger_likelihood_gb.meta.json").read_text()
            )
    except Exception:
        pass

    # ---------- return ----------
    return {
        "status": "ok",
        "rows": n_rows,
        "version_dir": str(version_dir),
        "logistic_meta": meta_lr,
        "rf_meta": rf_meta,
        "gb_meta": gb_meta,
    }


# ----------- CLI -----------
if __name__ == "__main__":
    out = retrain_from_log()
    # Print a compact one-line summary (useful in CI logs)
    status = out.get("status")
    rows = out.get("rows")
    if status == "skipped_low_sample":
        print(
            f"[retrain_from_log] SKIPPED: rows={rows} "
            f"(min={out.get('min_rows')}) — {out.get('message')}"
        )
    else:
        created_at = (
            (out.get("logistic_meta") or {}).get("created_at") or _utcnow_iso()
        )
        print(
            f"[retrain_from_log] OK: rows={rows}, created_at={created_at}, "
            f"version_dir={out.get('version_dir')}"
        )