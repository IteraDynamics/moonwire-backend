from __future__ import annotations
import json, os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

from src.ml.feature_builder import build_examples, synth_demo_dataset, FEATURE_ORDER
try:
    from src.paths import MODELS_DIR, LOGS_DIR
except Exception:
    MODELS_DIR, LOGS_DIR = Path("models"), Path("logs")

MODEL_NAME = "trigger_likelihood_v0"

def _prepare_xy(rows) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([r.x for r in rows], dtype=float)
    y = np.array([r.y for r in rows], dtype=int)
    return X, y

def train(days: int = 14, interval: str = "hour", out_dir: Path | None = None) -> Dict[str, Any]:
    out_dir = out_dir or MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, feat_order = build_examples(LOGS_DIR / "retraining_log.jsonl", LOGS_DIR / "retraining_triggered.jsonl", days=days, interval=interval)
    meta_extras = {}
    if not rows and os.getenv("DEMO_MODE", "false").lower() in ("1","true","yes"):
        rows, feat_order, meta_extras = synth_demo_dataset()

    if not rows:
        raise RuntimeError("No training rows and DEMO_MODE is false; cannot train.")

    rows.sort(key=lambda r: r.ts)  # time-aware split
    n = len(rows); split = max(int(n * 0.8), 1)
    tr, va = rows[:split], rows[split:]

    Xtr, ytr = _prepare_xy(tr)
    Xva, yva = _prepare_xy(va)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(Xtr, ytr)

    p_tr = clf.predict_proba(Xtr)[:, 1]
    p_va = clf.predict_proba(Xva)[:, 1] if len(va) > 0 else np.array([])

    metrics = {
        "roc_auc_tr": float(roc_auc_score(ytr, p_tr)),
        "pr_auc_tr": float(average_precision_score(ytr, p_tr)),
        "logloss_tr": float(log_loss(ytr, p_tr, labels=[0,1])),
        "brier_tr": float(brier_score_loss(ytr, p_tr)),
    }
    if len(va) > 0:
        metrics.update({
            "roc_auc_va": float(roc_auc_score(yva, p_va)),
            "pr_auc_va": float(average_precision_score(yva, p_va)),
            "logloss_va": float(log_loss(yva, p_va, labels=[0,1])),
            "brier_va": float(brier_score_loss(yva, p_va)),
        })

    model_path = out_dir / f"{MODEL_NAME}.joblib"
    joblib.dump(clf, model_path)

    meta = {
        "model_name": MODEL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", ""),
        "feature_order": feat_order,
        "metrics": metrics,
        "demo": bool(meta_extras.get("demo", False)),
    }
    (out_dir / f"{MODEL_NAME}.meta.json").write_text(json.dumps(meta, indent=2))

    return {"model_path": str(model_path), "meta": meta}

if __name__ == "__main__":
    # simple CLI
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--interval", type=str, default="hour")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()
    out = train(days=args.days, interval=args.interval, out_dir=Path(args.out_dir) if args.out_dir else None)
    print(json.dumps(out, indent=2))
