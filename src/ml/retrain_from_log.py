# src/ml/retrain_from_log.py
from __future__ import annotations

import json, os, math, shutil
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
)
from sklearn.model_selection import StratifiedKFold

from src import paths

# ----------------------- defaults / config -----------------------
TRAIN_LOG_PATH_DEFAULT = paths.MODELS_DIR / "training_data.jsonl"
MODELS_DIR = paths.MODELS_DIR

MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.5.0")
MIN_CV_ROWS = int(os.getenv("RETRAIN_MIN_CV_ROWS", "50"))  # below this → prefer CV metrics

# canonical filenames used elsewhere in the app
LOGI_MODEL_NAME = "trigger_likelihood_v0.joblib"
LOGI_META_NAME  = "trigger_likelihood_v0.meta.json"
RF_MODEL_NAME   = "trigger_likelihood_rf.joblib"
RF_META_NAME    = "trigger_likelihood_rf.meta.json"
GB_MODEL_NAME   = "trigger_likelihood_gb.joblib"
GB_META_NAME    = "trigger_likelihood_gb.meta.json"
COVERAGE_NAME   = "feature_coverage.json"

# ----------------------- util helpers -----------------------
def _git_sha() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

def _read_jsonl(path: Path) -> List[dict]:
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

def _clip_proba(p: np.ndarray) -> np.ndarray:
    if p is None or p.size == 0:
        return p
    return np.clip(p, 1e-6, 1.0 - 1e-6)

def _is_single_class(y: np.ndarray) -> bool:
    try:
        return np.unique(y).size < 2
    except Exception:
        return True

def _fmt_metric(v) -> str:
    try:
        if v is None:
            return "n/a"
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "n/a"
        return f"{float(v):.2f}"
    except Exception:
        return "n/a"

def _metrics_holdout(y_tr, p_tr, y_va, p_va) -> Dict[str, Any]:
    # keep logloss finite
    p_tr = _clip_proba(p_tr) if p_tr is not None else None
    p_va = _clip_proba(p_va) if p_va is not None else None

    def _safe(fn, yt, yp, default=None):
        try:
            if yt is None or yp is None or len(yt) == 0 or len(yp) == 0 or _is_single_class(yt):
                return default
            return float(fn(yt, yp))
        except Exception:
            return default

    return {
        "roc_auc_tr": _safe(roc_auc_score, y_tr, p_tr, None),
        "roc_auc_va": _safe(roc_auc_score, y_va, p_va, None),
        "pr_auc_tr":  _safe(average_precision_score, y_tr, p_tr, None),
        "pr_auc_va":  _safe(average_precision_score, y_va, p_va, None),
        "logloss_tr": _safe(log_loss, y_tr, p_tr, None),
        "logloss_va": _safe(log_loss, y_va, p_va, None),
    }

def _metrics_cv(model_ctor, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, Any]:
    if X.size == 0 or y.size == 0:
        return {"roc_auc_cv": None, "pr_auc_cv": None, "logloss_cv": None}

    uniq, counts = np.unique(y, return_counts=True)
    max_splits = int(np.min(counts)) if counts.size > 0 else 2
    n_splits = max(2, min(n_splits, max_splits))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs, prs, lls = [], [], []
    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        m = model_ctor()
        m.fit(Xtr, ytr)

        pva = m.predict_proba(Xva)[:, 1]
        pva = _clip_proba(pva)

        try:
            aucs.append(roc_auc_score(yva, pva))
        except Exception:
            pass
        try:
            prs.append(average_precision_score(yva, pva))
        except Exception:
            pass
        try:
            lls.append(log_loss(yva, pva))
        except Exception:
            pass

    def _mean_or_none(xs):
        xs = [x for x in xs if x is not None and not math.isnan(x) and not math.isinf(x)]
        return float(np.mean(xs)) if xs else None

    return {
        "roc_auc_cv": _mean_or_none(aucs),
        "pr_auc_cv":  _mean_or_none(prs),
        "logloss_cv": _mean_or_none(lls),
    }

def _top_coefficients(model: LogisticRegression, feat_order: List[str], top: int = 5) -> List[Dict[str, float]]:
    if not hasattr(model, "coef_") or model.coef_ is None:
        return []
    coef = model.coef_.ravel().tolist()
    pairs = list(zip(feat_order, coef))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "coef": float(v)} for k, v in pairs[:top]]

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

# ----------------------- data loading -----------------------
@dataclass
class TrainRow:
    features: Dict[str, float]
    label: int

def _load_training_rows(train_log_path: Path) -> List[TrainRow]:
    raw = _read_jsonl(train_log_path)
    rows: List[TrainRow] = []
    for r in raw:
        feats = r.get("features") or {}
        lbl = r.get("label")
        if isinstance(lbl, bool):
            y = 1 if lbl else 0
        else:
            try:
                y = int(lbl)
            except Exception:
                continue
        # coerce numerics
        fnum = {}
        for k, v in feats.items():
            try:
                fnum[k] = float(v)
            except Exception:
                fnum[k] = 0.0
        rows.append(TrainRow(features=fnum, label=y))
    return rows

def _matrix_from_rows(rows: List[TrainRow]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # unify feature space across rows
    keys = set()
    for r in rows:
        keys.update(r.features.keys())
    feat_order = sorted(keys)
    X = np.asarray([[float(r.features.get(k, 0.0) or 0.0) for k in feat_order] for r in rows], dtype=float)
    y = np.asarray([int(r.label) for r in rows], dtype=int)
    return X, y, feat_order

# ----------------------- main retrain -----------------------
def retrain_from_log(
    train_log_path: Path = TRAIN_LOG_PATH_DEFAULT,
    out_dir: Path = MODELS_DIR,
    version: str = MODEL_VERSION,
) -> Dict[str, Any]:
    rows = _load_training_rows(train_log_path)
    if not rows:
        raise RuntimeError(f"No rows in {train_log_path}; cannot retrain.")

    X, y, feat_order = _matrix_from_rows(rows)
    n = len(rows)
    uniq = np.unique(y)

    # simple 80/20 split (holdout) — but we may use CV metrics if data are tiny
    cut = max(1, int(0.8 * n))
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = (X[cut:], y[cut:]) if n > 1 else (X.copy(), y.copy())

    # ensure both classes present in train
    if _is_single_class(ytr) and not _is_single_class(y):
        # pull some opposite class from remainder
        other = 1 - int(ytr[0]) if ytr.size else 1
        idxs = np.where(y == other)[0]
        if idxs.size:
            take = idxs[: min(10, idxs.size)]
            Xtr = np.vstack([Xtr, X[take]])
            ytr = np.concatenate([ytr, y[take]])

    # coverage
    coverage = _compute_coverage_from_X(X, feat_order)

    # version dir
    ver_dir = out_dir / version
    ver_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Logistic ----------
    lr = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000)
    lr.fit(Xtr, ytr)
    p_tr_lr = lr.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va_lr = lr.predict_proba(Xva)[:, 1] if Xva.size else np.array([])

    use_cv_lr = (len(y) < MIN_CV_ROWS) or _is_single_class(yva)
    if not use_cv_lr:
        metrics_lr = _metrics_holdout(ytr, p_tr_lr, yva, p_va_lr)
    else:
        metrics_lr = _metrics_cv(
            lambda: LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000),
            X, y, n_splits=5
        )
    metrics_lr_serializable = {k: (_fmt_metric(v) if v is not None else "n/a") for k, v in metrics_lr.items()}

    # ---------- RF ----------
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(Xtr, ytr)
    p_tr_rf = rf.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va_rf = rf.predict_proba(Xva)[:, 1] if Xva.size else np.array([])

    use_cv_rf = (len(y) < MIN_CV_ROWS) or _is_single_class(yva)
    if not use_cv_rf:
        metrics_rf = _metrics_holdout(ytr, p_tr_rf, yva, p_va_rf)
    else:
        metrics_rf = _metrics_cv(
            lambda: RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
            ),
            X, y, n_splits=5
        )
    metrics_rf_serializable = {k: (_fmt_metric(v) if v is not None else "n/a") for k, v in metrics_rf.items()}

    # ---------- GB ----------
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    gb.fit(Xtr, ytr)
    p_tr_gb = gb.predict_proba(Xtr)[:, 1] if Xtr.size else np.array([])
    p_va_gb = gb.predict_proba(Xva)[:, 1] if Xva.size else np.array([])

    use_cv_gb = (len(y) < MIN_CV_ROWS) or _is_single_class(yva)
    if not use_cv_gb:
        metrics_gb = _metrics_holdout(ytr, p_tr_gb, yva, p_va_gb)
    else:
        metrics_gb = _metrics_cv(
            lambda: GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
            X, y, n_splits=5
        )
    metrics_gb_serializable = {k: (_fmt_metric(v) if v is not None else "n/a") for k, v in metrics_gb.items()}

    # ---------- write artifacts ----------
    # versioned paths
    v_logi_model = ver_dir / LOGI_MODEL_NAME
    v_logi_meta  = ver_dir / LOGI_META_NAME
    v_rf_model   = ver_dir / RF_MODEL_NAME
    v_rf_meta    = ver_dir / RF_META_NAME
    v_gb_model   = ver_dir / GB_MODEL_NAME
    v_gb_meta    = ver_dir / GB_META_NAME
    v_cov_path   = ver_dir / COVERAGE_NAME

    # top-level paths (back-compat)
    t_logi_model = MODELS_DIR / LOGI_MODEL_NAME
    t_logi_meta  = MODELS_DIR / LOGI_META_NAME
    t_rf_model   = MODELS_DIR / RF_MODEL_NAME
    t_rf_meta    = MODELS_DIR / RF_META_NAME
    t_gb_model   = MODELS_DIR / GB_MODEL_NAME
    t_gb_meta    = MODELS_DIR / GB_META_NAME
    t_cov_path   = MODELS_DIR / COVERAGE_NAME

    joblib.dump(lr, v_logi_model)
    joblib.dump(rf, v_rf_model)
    joblib.dump(gb, v_gb_model)

    with v_cov_path.open("w") as f:
        json.dump(coverage, f, indent=2)

    created_at = datetime.now(timezone.utc).isoformat()
    git_sha = _git_sha()

    meta_common = {
        "created_at": created_at,
        "git_sha": git_sha,
        "feature_order": feat_order,
        "demo": False,
        "version": version,
        "artifacts": {
            "model": "",  # set per model
            "feature_coverage": str(v_cov_path),
        },
        "display_metrics_note": f"metrics are {'CV' if len(y) < MIN_CV_ROWS else 'holdout/CV hybrid'}; small datasets may show 'n/a'",
        "feature_coverage_summary": {k: round(v.get("nonzero_pct", 0.0), 2) for k, v in coverage.items()},
    }

    # Logistic meta
    meta_lr = dict(meta_common)
    meta_lr["metrics"] = metrics_lr_serializable
    meta_lr["artifacts"] = dict(meta_common["artifacts"])
    meta_lr["artifacts"]["model"] = str(v_logi_model)
    meta_lr["top_features"] = _top_coefficients(lr, feat_order, top=5)
    with v_logi_meta.open("w") as f:
        json.dump(meta_lr, f, indent=2)

    # RF meta
    meta_rf = dict(meta_common)
    meta_rf["metrics"] = metrics_rf_serializable
    meta_rf["artifacts"] = dict(meta_common["artifacts"])
    meta_rf["artifacts"]["model"] = str(v_rf_model)
    with v_rf_meta.open("w") as f:
        json.dump(meta_rf, f, indent=2)

    # GB meta
    meta_gb = dict(meta_common)
    meta_gb["metrics"] = metrics_gb_serializable
    meta_gb["artifacts"] = dict(meta_common["artifacts"])
    meta_gb["artifacts"]["model"] = str(v_gb_model)
    meta_gb["top_features"] = []  # GB: not using feature importances here
    with v_gb_meta.open("w") as f:
        json.dump(meta_gb, f, indent=2)

    # copy to top-level filenames for back-compat
    shutil.copy2(v_logi_model, t_logi_model)
    shutil.copy2(v_logi_meta,  t_logi_meta)
    shutil.copy2(v_rf_model,   t_rf_model)
    shutil.copy2(v_rf_meta,    t_rf_meta)
    shutil.copy2(v_gb_model,   t_gb_model)
    shutil.copy2(v_gb_meta,    t_gb_meta)
    shutil.copy2(v_cov_path,   t_cov_path)

    return {
        "version": version,
        "n_rows": n,
        "n_pos": int(np.sum(y)),
        "n_neg": int(n - np.sum(y)),
        "feature_order": feat_order,
        "artifacts": {
            "version_dir": str(ver_dir),
            "logistic": {"model": str(v_logi_model), "meta": str(v_logi_meta)},
            "rf":       {"model": str(v_rf_model),   "meta": str(v_rf_meta)},
            "gb":       {"model": str(v_gb_model),   "meta": str(v_gb_meta)},
            "coverage": str(v_cov_path),
        },
        "metrics": {
            "logistic": metrics_lr_serializable,
            "rf": metrics_rf_serializable,
            "gb": metrics_gb_serializable,
        },
    }

# ----------------------- CLI -----------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Retrain TL models from training_data.jsonl")
    p.add_argument("--train-log", type=str, default=str(TRAIN_LOG_PATH_DEFAULT),
                   help="Path to training_data.jsonl")
    p.add_argument("--out-dir", type=str, default=str(MODELS_DIR),
                   help="Directory to write model artifacts")
    p.add_argument("--version", type=str, default=MODEL_VERSION,
                   help="Version string (e.g., v0.5.0)")
    args = p.parse_args()

    out = retrain_from_log(
        train_log_path=Path(args.train_log),
        out_dir=Path(args.out_dir),
        version=args.version,
    )

    print(json.dumps(out, indent=2))