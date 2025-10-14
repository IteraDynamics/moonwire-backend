# scripts/summary_sections/retrain_summary.py
from __future__ import annotations
import os, json
from pathlib import Path
from scripts.summary_sections.common import SummaryContext

def append(md: list[str], ctx: SummaryContext):
    md.append("\n### 🧪 Retrain Summary")
    try:
        version = os.getenv("MODEL_VERSION", "v0.5.0")
        vdir = ctx.models_dir / version

        td_path = ctx.models_dir / "training_data.jsonl"
        rows_cnt = 0
        if td_path.exists():
            try:
                rows_cnt = sum(1 for _ in td_path.open("r", encoding="utf-8"))
            except Exception:
                rows_cnt = 0
        md.append(f"rows={rows_cnt}")

        if not vdir.exists():
            md.append("\t- retrain skipped or no artifacts found")
            return

        metas = []
        for name, fname in [
            ("logistic", "trigger_likelihood_v0.meta.json"),
            ("rf",       "trigger_likelihood_rf.meta.json"),
            ("gb",       "trigger_likelihood_gb.meta.json"),
        ]:
            j = vdir / fname
            if j.exists():
                try:
                    with j.open("r", encoding="utf-8") as f:
                        metas.append((name, json.load(f)))
                except Exception:
                    pass

        if not metas:
            md.append("\t- retrain skipped or no artifacts found")
            return

        model_names = ", ".join(n for n, _ in metas)
        md.append(f"\t- Models: {model_names}")

        created = (metas[0][1] or {}).get("created_at")
        if created:
            md.append(f"\t- created_at={created}")

        def _fmt(x):
            if x is None: return "n/a"
            try: return f"{float(x):.2f}"
            except Exception: return str(x)

        for name, meta in metas:
            m = (meta or {}).get("metrics") or {}
            auc = m.get("roc_auc_va"); pr = m.get("pr_auc_va"); ll = m.get("logloss_va")
            md.append(f"\t- {name}: ROC-AUC={_fmt(auc)} | PR-AUC={_fmt(pr)} | LogLoss={_fmt(ll)}")
            cb = (meta or {}).get("class_balance") or {}
            if isinstance(cb, dict) and (cb.get(0) is not None or cb.get(1) is not None):
                pos = int(cb.get(1, 0)); neg = int(cb.get(0, 0))
                md.append(f"\t  - labels: pos={pos}, neg={neg}")
                if auc is None or pr is None:
                    md.append("\t  - ⚠️ insufficient label diversity for AUC (need both classes)")

        try:
            tf = (metas[0][1] or {}).get("top_features") or []
        except Exception:
            tf = []
        if tf:
            tops = ", ".join(t.get("feature","?") for t in tf[:3])
            md.append(f"\t- top features: {tops}")

    except Exception as e:
        md.append(f"⚠️ Retrain Summary failed: {e}")