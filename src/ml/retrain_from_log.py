# src/ml/retrain_from_log.py
from __future__ import annotations

import json, os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

from src.paths import MODELS_DIR

# ---------- tiny helpers copied/adapted from train_trigger_model ----------
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

def _git_sha() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

def _top_coefficients(model: LogisticRegression, feat_order: List[str], top: int = 5) -> List[Dict[str, float]]:
    if not hasattr(model, "coef_") or model.coef_ is None:
        return []
    coef = model.coef_.ravel().tolist()
    pairs = list(zip(feat_order, coef))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "coef": float(v)} for k, v in pairs[:top]]

# ---------- IO ----------
def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out

# ---------- core ----------
def _mk_Xy_from_training_log(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    rows: [{"features": {...}, "label": bool, "origin": "...", "timestamp": "...", ...}, ...]
    """
    if not rows:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=int), []

    # union feature set (stable order)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in (r.get("features") or {}).keys():
            if k not in seen:
                keys.append(k); seen.add(k)

    # build X, y
    X: List[List[float]] = []
    y: List[int] = []
    for r in rows:
        feats = r.get("features") or {}
        X.append([float(feats.get(k, 0.0) or 0.0) for k in keys])
        y.append(1 if bool(r.get("label")) else 0)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), keys

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
            "nonzero_pct": nonzero,
            "mean": float(np.mean(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return out

def retrain_from_log(
    *,
    train_log_path: Path | None = None,
    out_root: Path | None = None,
    version: str | None = None,
) -> Dict[str, Any]:
    """
    Retrain all models from models/training_data.jsonl (or provided path).
    Saves artifacts in models/<version>/ and also refreshes the non-versioned
    filenames for back-compat (trigger_likelihood_v0.*).
    """
    out_root = out_root or MODELS_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    train_log_path = train_log_path or (out_root / "training_data.jsonl")
    version = version or os.getenv("MODEL_VERSION", "v0.5.0")

    rows = _load_jsonl(train_log_path)
    if not rows:
        raise RuntimeError(f"No rows in {train_log_path}; cannot retrain.")

    X, y, feat_order = _mk_Xy_from_training_log(rows)
    n = X.shape[0]
    cut = max(1, int(0.8 * n))
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = (X[cut:], y[cut:]) if n > 1 else (X.copy(), y.copy())

    # ensure 2 classes in train
    if Xtr.size > 0 and np.unique(ytr).size < 2 and np.unique(y).size >= 2:
        other = 1 - int(ytr[0])
        idxs = np.where(y == other)[0]
        if idxs.size:
            take = idxs[: min(10, idxs.size)]
            Xtr = np.vstack([Xtr, X[take]])
            ytr = np.concatenate([ytr, y[take]])

    coverage = _coverage_from_X(X, feat_order)

    # versioned dir
    vdir = out_root / version
    vdir.mkdir(parents=True, exist_ok=True)

    # ---------- Logistic ----------
    lr = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000)
    lr.fit(Xtr, ytr)
    p_tr_lr = lr.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va_lr = lr.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    metrics_lr = _metrics_dict(ytr, p_tr_lr, yva, p_va_lr)

    # save
    lr_model_v = vdir / "trigger_likelihood_v0.joblib"
    lr_meta_v  = vdir / "trigger_likelihood_v0.meta.json"
    cov_v      = vdir / "feature_coverage.json"
    joblib.dump(lr, lr_model_v)
    cov_v.write_text(json.dumps(coverage, indent=2))

    meta_lr = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "version": version,
        "feature_order": feat_order,
        "metrics": metrics_lr,
        "demo": False,
        "artifacts": {
            "model": str(lr_model_v),
            "feature_coverage": str(cov_v),
        },
        "top_features": _top_coefficients(lr, feat_order, top=5),
        "feature_coverage_summary": {k: round(v.get("nonzero_pct", 0.0), 2) for k, v in coverage.items()},
        "source": {"type": "training_data.jsonl", "rows": n, "path": str(train_log_path)},
    }
    lr_meta_v.write_text(json.dumps(meta_lr, indent=2))

    # also refresh back-compat non-versioned filenames in models/
    joblib.dump(lr, out_root / "trigger_likelihood_v0.joblib")
    (out_root / "trigger_likelihood_v0.meta.json").write_text(json.dumps(meta_lr, indent=2))
    (out_root / "feature_coverage.json").write_text(json.dumps(coverage, indent=2))

    # ---------- RF ----------
    rf_trained = False
    metrics_rf = None
    try:
        if np.unique(y).size >= 2:
            rf = RandomForestClassifier(
                n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
            )
            rf.fit(Xtr, ytr)
            p_tr_rf = rf.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
            p_va_rf = rf.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
            metrics_rf = _metrics_dict(ytr, p_tr_rf, yva, p_va_rf)
            # save
            joblib.dump(rf, vdir / "trigger_likelihood_rf.joblib")
            (vdir / "trigger_likelihood_rf.meta.json").write_text(json.dumps({
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": _git_sha(),
                "version": version,
                "feature_order": feat_order,
                "metrics": metrics_rf,
                "demo": False,
                "artifacts": {
                    "model": str(vdir / "trigger_likelihood_rf.joblib"),
                    "feature_coverage": str(cov_v),
                },
                "source": {"type": "training_data.jsonl", "rows": n, "path": str(train_log_path)},
            }, indent=2))
            # back-compat
            joblib.dump(rf, MODELS_DIR / "trigger_likelihood_rf.joblib")
            (MODELS_DIR / "trigger_likelihood_rf.meta.json").write_text(json.dumps({
                "created_at": meta_lr["created_at"],
                "git_sha": meta_lr["git_sha"],
                "version": version,
                "feature_order": feat_order,
                "metrics": metrics_rf,
                "demo": False,
                "artifacts": {
                    "model": str(MODELS_DIR / "trigger_likelihood_rf.joblib"),
                    "feature_coverage": str(MODELS_DIR / "feature_coverage.json"),
                },
                "source": {"type": "training_data.jsonl", "rows": n, "path": str(train_log_path)},
            }, indent=2))
            rf_trained = True
    except Exception:
        rf_trained = False
        metrics_rf = None

    # ---------- GB ----------
    gb_trained = False
    metrics_gb = None
    try:
        if np.unique(y).size >= 2:
            gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
            gb.fit(Xtr, ytr)
            p_tr_gb = gb.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
            p_va_gb = gb.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
            metrics_gb = _metrics_dict(ytr, p_tr_gb, yva, p_va_gb)
            # save
            joblib.dump(gb, vdir / "trigger_likelihood_gb.joblib")
            (vdir / "trigger_likelihood_gb.meta.json").write_text(json.dumps({
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": _git_sha(),
                "version": version,
                "feature_order": feat_order,
                "metrics": metrics_gb,
                "demo": False,
                "artifacts": {
                    "model": str(vdir / "trigger_likelihood_gb.joblib"),
                    "feature_coverage": str(cov_v),
                },
                "source": {"type": "training_data.jsonl", "rows": n, "path": str(train_log_path)},
            }, indent=2))
            # back-compat (we only expose GB via ensemble anyway)
            joblib.dump(gb, MODELS_DIR / "trigger_likelihood_gb.joblib")
            (MODELS_DIR / "trigger_likelihood_gb.meta.json").write_text(json.dumps({
                "created_at": meta_lr["created_at"],
                "git_sha": meta_lr["git_sha"],
                "version": version,
                "feature_order": feat_order,
                "metrics": metrics_gb,
                "demo": False,
                "artifacts": {
                    "model": str(MODELS_DIR / "trigger_likelihood_gb.joblib"),
                    "feature_coverage": str(MODELS_DIR / "feature_coverage.json"),
                },
                "source": {"type": "training_data.jsonl", "rows": n, "path": str(train_log_path)},
            }, indent=2))
            gb_trained = True
    except Exception:
        gb_trained = False
        metrics_gb = None

    return {
        "version": version,
        "rows": n,
        "feature_order": feat_order,
        "log_path": str(train_log_path),
        "artifacts_dir": str(vdir),
        "metrics": {
            "logistic": metrics_lr,
            "rf": metrics_rf,
            "gb": metrics_gb,
        },
    }

# ---------- simple CLI ----------
if __name__ == "__main__":
    v = os.getenv("MODEL_VERSION", "v0.5.0")
    log_path = os.getenv("TRAIN_LOG_PATH")
    out = retrain_from_log(
        train_log_path=Path(log_path) if log_path else None,
        out_root=MODELS_DIR,
        version=v,
    )
    print(json.dumps(out, indent=2))