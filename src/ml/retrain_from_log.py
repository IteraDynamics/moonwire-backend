# src/ml/retrain_from_log.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# ---------- helpers ----------

def _git_sha() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _load_jsonl(path: Path) -> List[dict]:
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
            pass
    return out


def _collect_feature_order(rows: List[dict]) -> List[str]:
    seen = set()
    order: List[str] = []
    for r in rows:
        feats = r.get("features") or {}
        if not isinstance(feats, dict):
            continue
        for k in feats.keys():
            if k not in seen:
                seen.add(k)
                order.append(k)
    return order


def _mk_matrix(rows: List[dict], feat_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    for r in rows:
        feats = r.get("features") or {}
        row = [float(feats.get(k, 0.0) or 0.0) for k in feat_order]
        X.append(row)
        y.append(1 if bool(r.get("label", False)) else 0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


# --- Robust metrics helper: never throws; AUC only when both classes exist ---
def _safe_scores(y_true, p) -> Dict[str, Any]:
    """
    Return robust metrics + class balance. AUCs are only computed when both
    classes are present. LogLoss is clipped to avoid -inf/NaN on tiny sets.
    Output keys:
      - roc_auc (float|None)
      - pr_auc  (float|None)
      - logloss (float|None)
      - class_balance: {0: n0, 1: n1}
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

    out: Dict[str, Any] = {"roc_auc": None, "pr_auc": None, "logloss": None, "class_balance": {}}

    y = np.asarray(y_true)
    uniq, cnt = np.unique(y, return_counts=True)
    out["class_balance"] = {int(k): int(v) for k, v in zip(uniq, cnt)}

    # AUCs only valid if both classes present
    if len(uniq) >= 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, p))
        except Exception:
            out["roc_auc"] = None
        try:
            out["pr_auc"] = float(average_precision_score(y_true, p))
        except Exception:
            out["pr_auc"] = None

    # LogLoss (well-defined even with single class, with clipping)
    try:
        eps = 1e-9
        pp = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
        out["logloss"] = float(log_loss(y_true, pp, labels=[0, 1]))
    except Exception:
        out["logloss"] = None

    return out


def _top_coefficients(model: LogisticRegression, feat_order: List[str], top: int = 5) -> List[Dict[str, float]]:
    if not hasattr(model, "coef_") or model.coef_ is None:
        return []
    coef = model.coef_.ravel().tolist()
    pairs = list(zip(feat_order, coef))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "coef": float(v)} for k, v in pairs[:top]]


@dataclass
class SaveBundle:
    model_path: Path
    meta_path: Path


def _save_model_and_meta(
    model: Any,
    feat_order: List[str],
    metrics: Dict[str, Any],
    out_dir: Path,
    basename: str,
    top_features: List[Dict[str, float]] | None = None,
) -> SaveBundle:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{basename}.joblib"
    meta_path = out_dir / f"{basename}.meta.json"

    joblib.dump(model, model_path)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": {
            "roc_auc_va": metrics.get("roc_auc"),
            "pr_auc_va": metrics.get("pr_auc"),
            "logloss_va": metrics.get("logloss"),
        },
        "class_balance": metrics.get("class_balance", {}),
        "artifacts": {"model": str(model_path)},
    }
    if top_features is not None:
        meta["top_features"] = top_features

    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return SaveBundle(model_path=model_path, meta_path=meta_path)


# ---------- main retrain ----------

def retrain_from_log(
    train_log_path: Path,
    save_dir: Path,
    version: str = "v0.5.0",
    val_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    rows = _load_jsonl(train_log_path)
    if not rows:
        raise RuntimeError(f"No rows in {train_log_path}; cannot retrain.")

    feat_order = _collect_feature_order(rows)
    if not feat_order:
        raise RuntimeError("No features found in training rows.")

    X, y = _mk_matrix(rows, feat_order)

    # Split; stratify only if both classes present
    try:
        strat = y if len(np.unique(y)) >= 2 else None
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=val_size, random_state=random_state, stratify=strat
        )
    except Exception:
        # Fallback: no split if too few rows
        Xtr, ytr = X, y
        Xva, yva = X, y

    # --- train logistic ---
    lr = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000)
    lr.fit(Xtr, ytr)
    p_va_lr = lr.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    scores_lr = _safe_scores(yva, p_va_lr)
    top_lr = _top_coefficients(lr, feat_order, top=5)

    # --- train RF ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(Xtr, ytr)
    p_va_rf = rf.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    scores_rf = _safe_scores(yva, p_va_rf)

    # --- train GB ---
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    gb.fit(Xtr, ytr)
    p_va_gb = gb.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    scores_gb = _safe_scores(yva, p_va_gb)

    # --- save artifacts (versioned folder) ---
    version_dir = Path(save_dir) / version
    version_dir.mkdir(parents=True, exist_ok=True)

    lr_bundle = _save_model_and_meta(lr, feat_order, scores_lr, version_dir, "trigger_likelihood_v0", top_features=top_lr)
    rf_bundle = _save_model_and_meta(rf, feat_order, scores_rf, version_dir, "trigger_likelihood_rf")
    gb_bundle = _save_model_and_meta(gb, feat_order, scores_gb, version_dir, "trigger_likelihood_gb")

    return {
        "rows": len(rows),
        "version_dir": str(version_dir),
        "models": {
            "logistic": {"model": str(lr_bundle.model_path), "meta": str(lr_bundle.meta_path), "metrics": scores_lr},
            "rf": {"model": str(rf_bundle.model_path), "meta": str(rf_bundle.meta_path), "metrics": scores_rf},
            "gb": {"model": str(gb_bundle.model_path), "meta": str(gb_bundle.meta_path), "metrics": scores_gb},
        },
        "feature_order": feat_order,
    }


# ---------- CLI ----------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Retrain from labeled training_data.jsonl")
    ap.add_argument("--train-log", default=os.getenv("TRAIN_LOG_PATH", "models/training_data.jsonl"))
    ap.add_argument("--save-dir", default=os.getenv("SAVE_MODEL_DIR", "models"))
    ap.add_argument("--version", default=os.getenv("MODEL_VERSION", "v0.5.0"))
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = retrain_from_log(Path(args.train_log), Path(args.save_dir), version=args.version)
    print(json.dumps(out, indent=2))