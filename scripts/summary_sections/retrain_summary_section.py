# scripts/summary_sections/retrain_summary_section.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def _load_meta(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def render(md: List[str], models_root: Path | str = "models", env_var: str = "MODEL_VERSION") -> None:
    """
    Render the '🧪 Retrain Summary' section into `md`.

    Args:
        md: markdown lines list to append to.
        models_root: root directory containing versioned model folders (default: "models").
        env_var: environment variable that stores the version folder name (default: "MODEL_VERSION").
    """
    md.append("\n### 🧪 Retrain Summary")

    try:
        models_dir = Path(models_root)
        version = os.getenv(env_var, "v0.5.0")
        vdir = models_dir / version

        # Count rows in training_data.jsonl if present
        td_path = models_dir / "training_data.jsonl"
        rows_cnt = 0
        if td_path.exists():
            try:
                with td_path.open("r", encoding="utf-8") as f:
                    rows_cnt = sum(1 for _ in f)
            except Exception:
                rows_cnt = 0
        md.append(f"rows={rows_cnt}")

        if not vdir.exists():
            md.append("\t- retrain skipped or no artifacts found")
            return

        # Load metas if present (keep order: logistic, rf, gb)
        metas: List[Tuple[str, Dict[str, Any]]] = []
        for name, fname in [
            ("logistic", "trigger_likelihood_v0.meta.json"),
            ("rf",       "trigger_likelihood_rf.meta.json"),
            ("gb",       "trigger_likelihood_gb.meta.json"),
        ]:
            meta_path = vdir / fname
            meta = _load_meta(meta_path)
            if meta:
                metas.append((name, meta))

        if not metas:
            md.append("\t- retrain skipped or no artifacts found")
            return

        model_names = ", ".join(n for n, _ in metas)
        md.append(f"\t- Models: {model_names}")

        # show created_at once (from first meta we loaded)
        created = (metas[0][1] or {}).get("created_at")
        if created:
            md.append(f"\t- created_at={created}")

        # detail lines
        for name, meta in metas:
            m = (meta or {}).get("metrics") or {}
            auc = m.get("roc_auc_va")
            pr  = m.get("pr_auc_va")
            ll  = m.get("logloss_va")

            md.append(f"\t- {name}: ROC-AUC={_fmt(auc)} | PR-AUC={_fmt(pr)} | LogLoss={_fmt(ll)}")

            # class balance (if saved by retrainer)
            cb = (meta or {}).get("class_balance") or {}
            if isinstance(cb, dict) and (cb.get(0) is not None or cb.get(1) is not None):
                pos = int(cb.get(1, 0)); neg = int(cb.get(0, 0))
                md.append(f"\t  - labels: pos={pos}, neg={neg}")
                if auc is None or pr is None:
                    md.append("\t  - ⚠️ insufficient label diversity for AUC (need both classes)")

        # top features (from the first meta only, mirrors prior behavior)
        try:
            tf = (metas[0][1] or {}).get("top_features") or []
        except Exception:
            tf = []
        if tf:
            tops = ", ".join(t.get("feature", "?") for t in tf[:3])
            md.append(f"\t- top features: {tops}")

    except Exception as e:
        md.append(f"⚠️ Retrain Summary failed: {e}")