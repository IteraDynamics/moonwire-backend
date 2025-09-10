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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.paths import MODELS_DIR
from src.ml.training_metadata import save_training_metadata


# ------------------------- helpers -------------------------

@dataclass
class TrainArtifacts:
    model_path: Path
    meta_path: Path
    metrics: Dict[str, float]
    feature_order: List[str]
    top_features: List[Dict[str, float]] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> List[dict]:
    if not path or not path.exists():
        return []
    out: List[dict] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except Exception:
            # ignore bad lines
            pass
    return out


def _compute_feat_order(rows: List[dict]) -> List[str]:
    keys = set()
    for r in rows:
        fx = r.get("features") or {}
        for k in fx.keys():
            keys.add(str(k))
    # keep a stable deterministic order
    return sorted(keys)


def _mk_arrays(rows: List[dict], feat_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    for r in rows:
        f = r.get("features") or {}
        X.append([float(f.get(k, 0.0) or 0.0) for k in feat_order])
        y.append(1 if bool(r.get("label", False)) else 0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def _metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    def safe(fn, *args, default=0.0):
        try:
            return float(fn(*args))
        except Exception:
            return float(default)

    return {
        "roc_auc": safe(roc_auc_score, y_true, p, default=0.5),
        "pr_auc": safe(average_precision_score, y_true, p, default=0.0),
        "logloss": safe(log_loss, y_true, p, default=0.0),
    }


def _top_coefficients(model: LogisticRegression, feat_order: List[str], top: int = 5) -> List[Dict[str, float]]:
    if not hasattr(model, "coef_") or model.coef_ is None:
        return []
    coef = model.coef_.ravel().tolist()
    pairs = list(zip(feat_order, coef))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "coef": float(v)} for k, v in pairs[:top]]


def _write_meta(path: Path, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _git_sha() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


# ------------------------- training core -------------------------

def _train_logistic(
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, out_dir: Path, feat_order: List[str]
) -> TrainArtifacts:
    model = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
    )
    model.fit(Xtr, ytr)
    p_va = model.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    m = _metrics(yva, p_va)

    model_path = out_dir / "trigger_likelihood_v0.joblib"
    meta_path = out_dir / "trigger_likelihood_v0.meta.json"
    joblib.dump(model, model_path)

    meta = {
        "created_at": _now_iso(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": {
            "roc_auc_va": m["roc_auc"],
            "pr_auc_va": m["pr_auc"],
            "logloss_va": m["logloss"],
        },
    }
    _write_meta(meta_path, meta)
    return TrainArtifacts(model_path, meta_path, m, feat_order, _top_coefficients(model, feat_order))


def _train_rf(
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, out_dir: Path, feat_order: List[str]
) -> TrainArtifacts:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr)
    p_va = model.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    m = _metrics(yva, p_va)

    model_path = out_dir / "trigger_likelihood_rf.joblib"
    meta_path = out_dir / "trigger_likelihood_rf.meta.json"
    joblib.dump(model, model_path)

    meta = {
        "created_at": _now_iso(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": {
            "roc_auc_va": m["roc_auc"],
            "pr_auc_va": m["pr_auc"],
            "logloss_va": m["logloss"],
        },
    }
    _write_meta(meta_path, meta)
    return TrainArtifacts(model_path, meta_path, m, feat_order, None)


def _train_gb(
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, out_dir: Path, feat_order: List[str]
) -> TrainArtifacts:
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(Xtr, ytr)
    p_va = model.predict_proba(Xva)[:, 1] if Xva.size else np.array([])
    m = _metrics(yva, p_va)

    model_path = out_dir / "trigger_likelihood_gb.joblib"
    meta_path = out_dir / "trigger_likelihood_gb.meta.json"
    joblib.dump(model, model_path)

    meta = {
        "created_at": _now_iso(),
        "git_sha": _git_sha(),
        "feature_order": feat_order,
        "metrics": {
            "roc_auc_va": m["roc_auc"],
            "pr_auc_va": m["pr_auc"],
            "logloss_va": m["logloss"],
        },
    }
    _write_meta(meta_path, meta)
    return TrainArtifacts(model_path, meta_path, m, feat_order, None)


# ------------------------- public API -------------------------

def retrain_from_log(
    train_log_path: Path | None = None,
    save_dir: Path | None = None,
    version: str | None = None,
) -> Dict[str, Any]:
    """
    Retrain ensemble models from models/training_data.jsonl and write versioned artifacts.
    Also stamps models/training_version.txt with the current version for inference.
    """
    train_log = train_log_path or (MODELS_DIR / "training_data.jsonl")
    out_root = save_dir or MODELS_DIR
    ver = (version or os.getenv("MODEL_VERSION") or "v0.5.0").strip()
    ver_dir = out_root / ver
    ver_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(train_log)
    if not rows:
        raise RuntimeError(f"No rows in {train_log}; cannot retrain.")

    # counts for metadata snapshot
    origin_counts: Dict[str, int] = {}
    label_counts = {"true": 0, "false": 0}
    for r in rows:
        o = str(r.get("origin", "unknown")).strip().lower()
        origin_counts[o] = origin_counts.get(o, 0) + 1
        if bool(r.get("label", False)):
            label_counts["true"] += 1
        else:
            label_counts["false"] += 1

    feat_order = _compute_feat_order(rows)
    X, y = _mk_arrays(rows, feat_order)

    # guard tiny dataset
    if X.shape[0] < 3:
        # too small even for a split — train on all, eval equals train
        Xtr, ytr, Xva, yva = X, y, X, y
    else:
        try:
            Xtr, Xva, ytr, yva = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
        except Exception:
            Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

    # subdir for this version
    L_art = _train_logistic(Xtr, ytr, Xva, yva, ver_dir, feat_order)
    RF_art = _train_rf(Xtr, ytr, Xva, yva, ver_dir, feat_order)
    GB_art = _train_gb(Xtr, ytr, Xva, yva, ver_dir, feat_order)

    # also write "latest" (back-compat for existing inference code that looks for non-versioned names)
    for src, dst in [
        (L_art.model_path, MODELS_DIR / "trigger_likelihood_v0.joblib"),
        (L_art.meta_path,  MODELS_DIR / "trigger_likelihood_v0.meta.json"),
        (RF_art.model_path, MODELS_DIR / "trigger_likelihood_rf.joblib"),
        (RF_art.meta_path,  MODELS_DIR / "trigger_likelihood_rf.meta.json"),
        (GB_art.model_path, MODELS_DIR / "trigger_likelihood_gb.joblib"),
        (GB_art.meta_path,  MODELS_DIR / "trigger_likelihood_gb.meta.json"),
    ]:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            # copy bytes (avoid symlink surprises in CI)
            with src.open("rb") as fsrc, dst.open("wb") as fdst:
                fdst.write(fsrc.read())
        except Exception:
            pass

    # --- persist "current version" for inference ---
    try:
        version_file = (save_dir or MODELS_DIR) / "training_version.txt"
        version_file.parent.mkdir(parents=True, exist_ok=True)
        version_file.write_text(ver + "\n", encoding="utf-8")
    except Exception:
        # never fail retrain on version stamp write
        pass

    # ---- compact summary (returned) ----
    summary = {
        "version": ver,
        "rows": int(X.shape[0]),
        "origin_counts": origin_counts,
        "label_counts": label_counts,
        "logistic": {
            "model": str(L_art.model_path),
            "meta": str(L_art.meta_path),
            "metrics": L_art.metrics,
            "top_features": L_art.top_features or [],
        },
        "rf": {
            "model": str(RF_art.model_path),
            "meta": str(RF_art.meta_path),
            "metrics": RF_art.metrics,
        },
        "gb": {
            "model": str(GB_art.model_path),
            "meta": str(GB_art.meta_path),
            "metrics": GB_art.metrics,
        },
    }

    # ---- append a training-run snapshot for provenance (new in v0.5.1) ----
    try:
        metrics_block = {
            "logistic": {
                "roc_auc": float(L_art.metrics.get("roc_auc", 0.0)),
                "pr_auc": float(L_art.metrics.get("pr_auc", 0.0)),
                "logloss": float(L_art.metrics.get("logloss", 0.0)),
            },
            "rf": {
                "roc_auc": float(RF_art.metrics.get("roc_auc", 0.0)),
                "pr_auc": float(RF_art.metrics.get("pr_auc", 0.0)),
                "logloss": float(RF_art.metrics.get("logloss", 0.0)),
            },
            "gb": {
                "roc_auc": float(GB_art.metrics.get("roc_auc", 0.0)),
                "pr_auc": float(GB_art.metrics.get("pr_auc", 0.0)),
                "logloss": float(GB_art.metrics.get("logloss", 0.0)),
            },
        }
        top_feats_list = [tf["feature"] for tf in (L_art.top_features or [])][:5]
        save_training_metadata(
            version=ver,
            rows=int(X.shape[0]),
            origin_counts=origin_counts,
            label_counts=label_counts,
            metrics=metrics_block,
            top_features=top_feats_list,
        )
    except Exception as e:
        print(f"[training-metadata] skipped: {type(e).__name__}: {e}")

    return summary


# ------------------------- CLI -------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain ensemble models from training_data.jsonl")
    p.add_argument("--train-log", type=str, default=os.getenv("TRAIN_LOG_PATH", str(MODELS_DIR / "training_data.jsonl")))
    p.add_argument("--save-dir", type=str, default=os.getenv("SAVE_MODEL_DIR", str(MODELS_DIR)))
    p.add_argument("--version",  type=str, default=os.getenv("MODEL_VERSION", "v0.5.0"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = retrain_from_log(
        train_log_path=Path(args.train_log),
        save_dir=Path(args.save_dir),
        version=args.version,
    )
    print(json.dumps(summary, indent=2))