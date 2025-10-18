# scripts/ml/model_runner.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

def _get_model(model_type: str):
    if model_type == "gb":
        return GradientBoostingClassifier(random_state=42)
    return LogisticRegression(max_iter=1000, class_weight="balanced")

def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                model_type: str = "logreg"):
    model = _get_model(model_type)
    model.fit(X_train, y_train)
    return model

def predict_proba(model, X: np.ndarray) -> np.ndarray:
    # return p(long)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # fallback for models without predict_proba
    scores = model.decision_function(X)
    # sigmoid
    return 1 / (1 + np.exp(-scores))