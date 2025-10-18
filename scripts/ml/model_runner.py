# scripts/ml/model_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Core sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Try to use the faster, regularized histogram GB if available
try:
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    _HGB_AVAILABLE = True
except Exception:
    HistGradientBoostingClassifier = None  # type: ignore
    _HGB_AVAILABLE = False


# ---------- Internal helpers ----------

def _make_logreg():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=None,       # keep stable for CI
        random_state=42,
    )


def _make_gb():
    # Reasonable defaults; small learning rate, shallow trees
    return GradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=3,
        n_estimators=200,
        random_state=42,
    )


def _make_hgb():
    """
    More regularized, fast GB variant. Falls back to GB if HGB isn't available.
    """
    if not _HGB_AVAILABLE:
        return _make_gb()
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_leaf_nodes=63,
        min_samples_leaf=10,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )


@dataclass
class HybridEnsemble:
    """
    Simple probability-blend ensemble:
      p = alpha * p_tree + (1 - alpha) * p_linear
    where tree = HGB if available, else GB.
    """
    alpha: float = 0.6
    tree_model: Optional[object] = None
    linear_model: Optional[object] = None

    def __post_init__(self):
        if self.tree_model is None:
            self.tree_model = _make_hgb()
        if self.linear_model is None:
            self.linear_model = _make_logreg()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree_model.fit(X, y)
        self.linear_model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_tree = _predict_proba_safe(self.tree_model, X)
        p_lin  = _predict_proba_safe(self.linear_model, X)
        p = self.alpha * p_tree + (1.0 - self.alpha) * p_lin
        # Return a 2-col array to mimic sklearn's predict_proba shape
        return np.column_stack([1.0 - p, p])


def _predict_proba_safe(model, X: np.ndarray) -> np.ndarray:
    """
    Return the positive-class probability as a 1-D array.
    Works for classifiers with predict_proba or decision_function.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # proba shape (n,2) assumed; take column 1
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        # binary-but-single-column edge case
        if proba.ndim == 2 and proba.shape[1] == 1:
            return proba[:, 0]
        # fallback: best effort
        return np.squeeze(proba)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # logistic link
        return 1.0 / (1.0 + np.exp(-scores))
    # ultra-fallback: predict labels then cast to prob
    preds = model.predict(X)
    return preds.astype(float)


# ---------- Public API (unchanged signatures) ----------

def _get_model(model_type: str):
    mt = (model_type or "logreg").lower()
    if mt == "gb":
        return _make_gb()
    if mt == "hgb":
        return _make_hgb()
    if mt == "hybrid":
        # alpha can be tuned later; keep conservative blend
        return HybridEnsemble(alpha=0.6)
    # default
    return _make_logreg()


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model_type: str = "logreg",
):
    """
    Fit and return the model. Interface compatible with existing tests.
    """
    model = _get_model(model_type)
    model.fit(X_train, y_train)
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """
    Return p(long) as a 1-D vector, regardless of the underlying model.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            # binary: take positive class
            if proba.shape[1] >= 2:
                return proba[:, 1]
            return np.squeeze(proba)
        return np.squeeze(proba)
    # fallback for models without predict_proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = model.predict(X)
    return preds.astype(float)