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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

from src.paths import (
    LOGS_DIR,
    MODELS_DIR,
    RETRAINING_LOG_PATH,
    RETRAINING_TRIGGERED_LOG_PATH,
)
from src.ml.feature_builder import build_examples, synth_demo_dataset


LR_NAME = "trigger_likelihood_v0"
RF_NAME = "trigger_likelihood_rf"
GB_NAME = "trigger_likelihood_gb"
COVERAGE_NAME = "feature_coverage.json"


@dataclass
class TrainResult:
    model_path: str
    meta_path: str
    coverage_path: str | None
    metrics: Dict[str, Any]
    demo: bool
    rf_model_path: str | None = None
    rf_meta_path: str | None = None
    gb_model_path: str | None = None
    gb_meta_path: str | None = None


# ----------------- helpers -----------------
def _mk_arrays(rows: List[Dict[str, Any]], feat_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    # rows may be dicts or small dataclasses with attributes x/y; handle both
    for r in rows:
        if isinstance(r, dict):
            f = r.get("features", {}) or {}
            X.append([float(f.get(k, 0.0) or 0.0) for k in feat_order])
            y.append(int(r.get("label", 0)))
        else:
            # FeatureRow(ts, origin, x, y) from builder
            X.append([float(v) for v in getattr(r, "x")])
            y.append(int(getattr(r, "y")))
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


def _calc_metrics(ytr, p_tr, yva, p_va) -> Dict[str, Any]:
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


def _top_features_from_model(model, feat_order: List[str], top: int = 5) -> List[Dict[str, float]]:
    """Return top weights/importances by absolute magnitude."""
    try:
        if hasattr(model, "coef_") and model.coef_ is not None:
            coef = model.coef_.ravel().tolist()
            pairs = list(zip(feat_order, coef))
            pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
            return [{"feature": k, "coef": float(v)} for k, v in pairs[:top]]
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_.tolist()
            pairs = list(zip(feat_order, imp))
            pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
            return [{"feature": k, "importance": float(v)} for k, v in pairs[:top]]
    except Exception:
        pass
    return []


def _git_sha() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


# ----------------- training -----------------
def train(days: int = 14, interval: str = "hour", out_dir: Path | None = None) -> Dict[str, Any]:
    """
    Train logistic + random forest + gradient boosting (when data exists or DEMO_MODE is on).
    Persists artifacts for each model; coverage json is written once.
    Returns a dict containing at least the logistic paths for back-compat with tests.
    """
    out_dir = out_dir or MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build examples using live paths (monkeypatch-friendly)
    rows, feat_order = build_examples(
        RETRAINING_LOG_PATH,
        RETRAINING_TRIGGERED_LOG_PATH,
        days=days,
        interval=interval,
    )

    demo_used = False
    if not rows and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        rows, feat_order, _ = synth_demo_dataset()
        demo_used = True

    if not rows:
        raise RuntimeError("No training rows and DEMO_MODE is false; cannot train.")

    # Arrays + simple time-aware split
    X, y = _mk_arrays(rows, feat_order)
    n = X.shape[0]
    cut = max(1, int(0.8 * n))
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = X[cut:], y[cut:] if n > 1 else (X[:], y[:])

    # Compute and save coverage (once)
    coverage = _compute_coverage_from_X(X, feat_order)
    coverage_path = out_dir / COVERAGE_NAME
    with coverage_path.open("w") as f:
        json.dump(coverage, f, indent=2)

    created_at = datetime.now(timezone.utc).isoformat()
    sha = _git_sha()

    # ---- Logistic Regression
    lr = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
    )
    lr.fit(Xtr, ytr)
    p_tr = lr.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va = lr.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    lr_metrics = _calc_metrics(ytr, p_tr, yva, p_va)

    lr_model_path = out_dir / f"{LR_NAME}.joblib"
    lr_meta_path = out_dir / f"{LR_NAME}.meta.json"
    joblib.dump(lr, lr_model_path)
    lr_meta = {
        "created_at": created_at,
        "git_sha": sha,
        "feature_order": feat_order,
        "metrics": lr_metrics,
        "demo": bool(demo_used),
        "artifacts": {"model": str(lr_model_path), "feature_coverage": str(coverage_path)},
        "top_features": _top_features_from_model(lr, feat_order, top=5),
        "feature_coverage_summary": {k: round(v.get("nonzero_pct", 0.0), 2) for k, v in coverage.items()},
    }
    with lr_meta_path.open("w") as f:
        json.dump(lr_meta, f, indent=2)

    # ---- Random Forest
    rf_model_path = rf_meta_path = None
    try:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        )
        rf.fit(Xtr, ytr)
        p_tr = rf.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
        p_va = rf.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
        rf_metrics = _calc_metrics(ytr, p_tr, yva, p_va)

        rf_model_path = out_dir / f"{RF_NAME}.joblib"
        rf_meta_path = out_dir / f"{RF_NAME}.meta.json"
        joblib.dump(rf, rf_model_path)
        rf_meta = {
            "created_at": created_at,
            "git_sha": sha,
            "feature_order": feat_order,
            "metrics": rf_metrics,
            "demo": bool(demo_used),
            "artifacts": {"model": str(rf_model_path), "feature_coverage": str(coverage_path)},
            "top_features": _top_features_from_model(rf, feat_order, top=5),
            "feature_coverage_summary": {k: round(v.get("nonzero_pct", 0.0), 2) for k, v in coverage.items()},
        }
        with rf_meta_path.open("w") as f:
            json.dump(rf_meta, f, indent=2)
    except Exception:
        pass  # keep going; ensemble can still function

    # ---- Gradient Boosting (uses sample_weight for imbalance)
    gb_model_path = gb_meta_path = None
    try:
        gb = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        # balanced sample weights
        classes, counts = np.unique(ytr, return_counts=True)
        class_weights = {c: (len(ytr) / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
        sw = np.array([class_weights[int(c)] for c in ytr], dtype=float)

        gb.fit(Xtr, ytr, sample_weight=sw)
        p_tr = gb.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
        p_va = gb.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
        gb_metrics = _calc_metrics(ytr, p_tr, yva, p_va)

        gb_model_path = out_dir / f"{GB_NAME}.joblib"
        gb_meta_path = out_dir / f"{GB_NAME}.meta.json"
        joblib.dump(gb, gb_model_path)
        gb_meta = {
            "created_at": created_at,
            "git_sha": sha,
            "feature_order": feat_order,
            "metrics": gb_metrics,
            "demo": bool(demo_used),
            "artifacts": {"model": str(gb_model_path), "feature_coverage": str(coverage_path)},
            "top_features": _top_features_from_model(gb, feat_order, top=5),
            "feature_coverage_summary": {k: round(v.get("nonzero_pct", 0.0), 2) for k, v in coverage.items()},
        }
        with gb_meta_path.open("w") as f:
            json.dump(gb_meta, f, indent=2)
    except Exception:
        pass

    return {
        "model_path": str(lr_model_path),
        "meta_path": str(lr_meta_path),
        "coverage_path": str(coverage_path),
        "metrics": lr_metrics,
        "demo": demo_used,
        "rf_model_path": str(rf_model_path) if rf_model_path else None,
        "rf_meta_path": str(rf_meta_path) if rf_meta_path else None,
        "gb_model_path": str(gb_model_path) if gb_model_path else None,
        "gb_meta_path": str(gb_meta_path) if gb_meta_path else None,
    }


# --------------- tiny CLI for CI ----------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--interval", type=str, default="hour")
    args = p.parse_args()

    out = train(days=args.days, interval=args.interval, out_dir=MODELS_DIR)
    print(json.dumps(out, indent=2))
