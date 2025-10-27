# src/ml/train_trigger_ensemble.py
from __future__ import annotations

import json, os, math, random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

from src.paths import (
    MODELS_DIR,
    RETRAINING_LOG_PATH,
    RETRAINING_TRIGGERED_LOG_PATH,
    ML_HYPERPARAMETERS_PATH,
)
from src.ml.feature_builder import build_examples, synth_demo_dataset
import logging

logger = logging.getLogger(__name__)


@dataclass
class _ModelPack:
    key: str
    model: Any
    path: Path
    metrics: Dict[str, float]


def _mk_arrays(rows: List[Dict[str, Any]], feat_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for r in rows:
        f = r.get("features", {}) or {}
        X.append([float(f.get(k, 0.0) or 0.0) for k in feat_order])
        y.append(int(r.get("label", 0)))
    return np.asarray(X, float), np.asarray(y, int)


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


def _load_hyperparameters() -> Dict[str, Any]:
    """
    Load ML hyperparameters from config file.
    Returns defaults if file doesn't exist or is invalid.
    """
    default_config = {
        "ensemble": {
            "models": ["lr", "rf", "gb"],
            "weights": {"lr": 0.3, "rf": 0.4, "gb": 0.3}
        },
        "lr": {
            "solver": "liblinear",
            "max_iter": 2000,
            "class_weight": "balanced"
        },
        "rf": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        },
        "gb": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42
        },
        "training": {
            "days_lookback": 14,
            "interval": "hour",
            "train_val_split": 0.8
        }
    }

    try:
        if ML_HYPERPARAMETERS_PATH.exists():
            with ML_HYPERPARAMETERS_PATH.open("r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info(f"Loaded ML hyperparameters from {ML_HYPERPARAMETERS_PATH}")
                return config
    except Exception as e:
        logger.warning(f"Failed to load ML hyperparameters from {ML_HYPERPARAMETERS_PATH}: {e}. Using defaults.")

    return default_config


def _train_one(
    key: str, cls, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, hyperparams: Dict[str, Any]
) -> _ModelPack:
    """Train a single model with hyperparameters from config."""
    # Get model-specific hyperparameters
    model_params = hyperparams.get(key, {})

    if key == "lr":
        model = LogisticRegression(**model_params)
    elif key == "rf":
        model = RandomForestClassifier(**model_params)
    elif key == "gb":
        model = GradientBoostingClassifier(**model_params)
    else:
        model = cls()

    model.fit(Xtr, ytr)

    p_tr = model.predict_proba(Xtr)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Xtr)
    p_va = model.predict_proba(Xva)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Xva)

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
    return _ModelPack(key=key, model=model, path=Path(), metrics=metrics)


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in weights.values())
    if s <= 0:
        # fallback equal
        n = max(1, len(weights))
        return {k: 1.0 / n for k in weights}
    return {k: float(max(0.0, v) / s) for k, v in weights.items()}


def train_ensemble(
    days: int | None = None,
    interval: str | None = None,
    out_dir: Path | None = None,
    n_boot: int = 30,
) -> Dict[str, Any]:
    """
    Trains LR, RF, GB on the same features and writes:
      - trigger_ensemble_lr.joblib
      - trigger_ensemble_rf.joblib
      - trigger_ensemble_gb.joblib
      - trigger_ensemble.meta.json (with feature_order, weights, metrics, bootstrap bands)

    If days/interval are None, uses values from ml_hyperparameters.json config.
    """
    # Load hyperparameters config
    hyperparams = _load_hyperparameters()
    training_config = hyperparams.get("training", {})

    # Use config values if not explicitly provided
    days = days if days is not None else training_config.get("days_lookback", 14)
    interval = interval if interval is not None else training_config.get("interval", "hour")
    train_val_split = training_config.get("train_val_split", 0.8)

    out = out_dir or MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError("No training rows and DEMO_MODE is false; cannot train ensemble.")

    X, y = _mk_arrays(rows, feat_order)
    n = X.shape[0]
    cut = max(2, int(train_val_split * n))
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = X[cut:], y[cut:]

    packs: List[_ModelPack] = []
    # Train LR / RF / GB with hyperparameters from config
    packs.append(_train_one("lr", LogisticRegression, Xtr, ytr, Xva, yva, hyperparams))
    packs.append(_train_one("rf", RandomForestClassifier, Xtr, ytr, Xva, yva, hyperparams))
    packs.append(_train_one("gb", GradientBoostingClassifier, Xtr, ytr, Xva, yva, hyperparams))

    # Save models
    for p in packs:
        p.path = out / f"trigger_ensemble_{p.key}.joblib"
        joblib.dump(p.model, p.path)

    # Build validation ensemble probs for bootstrap
    def _proba(model, X_):
        return model.predict_proba(X_)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_)

    P = {p.key: _proba(p.model, Xva) if Xva.size else np.array([]) for p in packs}

    # Weighting by PR-AUC (more robust with class imbalance), fallback equal
    pr_auc = {p.key: float(max(0.0, p.metrics.get("pr_auc_va", 0.0))) for p in packs}
    weights = _normalize_weights(pr_auc)

    # Bootstrap: resample validation rows to estimate variance of the ensemble mean prob
    rng = random.Random(42)
    boot_means: List[float] = []
    if Xva.shape[0] > 0:
        idx_all = list(range(Xva.shape[0]))
        for _ in range(n_boot):
            idx = [rng.choice(idx_all) for _ in idx_all]  # sample size == len(idx_all)
            ens = 0.0
            for key, w in weights.items():
                if P[key].size:
                    ens += w * float(np.mean(P[key][idx]))
            boot_means.append(ens)

    if not boot_means:
        boot_mean = 0.5
        boot_std = 0.1
    else:
        boot_mean = float(np.mean(boot_means))
        boot_std = float(np.std(boot_means))

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "demo": bool(demo_used),
        "models": {
            p.key: {
                "path": str(p.path),
                "metrics": p.metrics,
            } for p in packs
        },
        "weights": weights,  # used at inference-time
        "bootstrap": {
            "n": n_boot,
            "mean": boot_mean,
            "std": boot_std,  # we’ll use this as a global band width heuristic
        },
    }

    meta_path = out / "trigger_ensemble.meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "meta_path": str(meta_path),
        "model_paths": [str(p.path) for p in packs],
        "weights": weights,
        "bootstrap_std": meta["bootstrap"]["std"],
        "demo": demo_used,
    }
