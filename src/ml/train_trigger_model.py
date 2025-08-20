from __future__ import annotations
import json, os, random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

from src.ml.feature_builder import build_examples, synth_demo_dataset
import src.paths as paths  # use live module attributes so monkeypatch works

MODEL_NAME = "trigger_likelihood_v0"


def _prepare_xy(rows) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([r.x for r in rows], dtype=float)
    y = np.array([r.y for r in rows], dtype=int)
    return X, y


def train(days: int = 14, interval: str = "hour", out_dir: Path | None = None) -> Dict[str, Any]:
    # Use paths.MODELS_DIR unless an explicit out_dir was provided
    out_dir = out_dir or paths.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build example paths from the CURRENT LOGS_DIR (monkeypatch-friendly)
    flags_path = paths.LOGS_DIR / "retraining_log.jsonl"
    triggers_path = paths.LOGS_DIR / "retraining_triggered.jsonl"

    rows, feat_order = build_examples(flags_path, triggers_path, days=days, interval=interval)
    meta_extras: Dict[str, Any] = {}

    # If no rows, allow DEMO_MODE to seed
    if not rows and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        rows, feat_order, meta_extras = synth_demo_dataset()

    # If still no rows → hard fail (keeps behavior when DEMO_MODE is off)
    if not rows:
        raise RuntimeError("No training rows and DEMO_MODE is false; cannot train.")

    # --- Guard 1: single-class dataset → augment minimally with demo rows
    y_all = np.array([r.y for r in rows], dtype=int)
    if len(np.unique(y_all)) < 2:
        demo_rows, _, _ = synth_demo_dataset()
        random.seed(42)
        # Pick a small deterministic slice spread across time
        # Take every 10th sample from the first ~300 demo rows to include some positives
        demo_slice = [demo_rows[i] for i in range(0, min(300, len(demo_rows)), 10)]
        rows = rows + demo_slice
        meta_extras["augmented_single_class"] = True

    # Time-aware split (sort by timestamp)
    rows.sort(key=lambda r: r.ts)
    n = len(rows)
    split = max(int(n * 0.8), 1)

    # --- Guard 2: if training fold is still single-class, slide the split forward
    def has_two_classes(arr: np.ndarray) -> bool:
        return len(np.unique(arr)) >= 2

    y_all = np.array([r.y for r in rows], dtype=int)
    if not has_two_classes(y_all[:split]) and has_two_classes(y_all):
        # Move split boundary forward until the train fold contains both classes,
        # while preserving time order. Cap at n-1 to keep a non-empty val fold.
        while split < n - 1 and not has_two_classes(y_all[:split]):
            split += 1
        meta_extras["split_adjusted_for_class_balance"] = True

    tr, va = rows[:split], rows[split:]

    Xtr, ytr = _prepare_xy(tr)
    if len(va) > 0:
        Xva = np.array([r.x for r in va], dtype=float)
        yva = np.array([r.y for r in va], dtype=int)
    else:
        Xva = np.empty((0, Xtr.shape[1])); yva = np.empty((0,), dtype=int)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(Xtr, ytr)

    p_tr = clf.predict_proba(Xtr)[:, 1]
    metrics = {
        "roc_auc_tr": float(roc_auc_score(ytr, p_tr)),
        "pr_auc_tr": float(average_precision_score(ytr, p_tr)),
        "logloss_tr": float(log_loss(ytr, p_tr, labels=[0, 1])),
        "brier_tr": float(brier_score_loss(ytr, p_tr)),
    }
    if len(va) > 0:
        p_va = clf.predict_proba(Xva)[:, 1]
        metrics.update({
            "roc_auc_va": float(roc_auc_score(yva, p_va)),
            "pr_auc_va": float(average_precision_score(yva, p_va)),
            "logloss_va": float(log_loss(yva, p_va, labels=[0, 1])),
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
        "notes": [k for k, v in meta_extras.items() if v],
    }
    (out_dir / f"{MODEL_NAME}.meta.json").write_text(json.dumps(meta, indent=2))

    return {"model_path": str(model_path), "meta": meta}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--interval", type=str, default="hour")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()
    out = train(days=args.days, interval=args.interval, out_dir=Path(args.out_dir) if args.out_dir else None)
    print(json.dumps(out, indent=2))