# scripts/summary_sections/latest_training_metadata.py
from __future__ import annotations
from math import isnan
from scripts.summary_sections.common import SummaryContext

def append(md: list[str], ctx: SummaryContext):
    md.append("\n📦 Latest Training Metadata")
    try:
        from src.ml import training_metadata
        latest = training_metadata.load_latest_training_metadata()
    except Exception as e:
        md.append(f"⚠️ Failed to load training metadata: {e}")
        return

    if not latest:
        md.append("No training metadata available (yet).")
        return

    version = latest.get("version", "n/a")
    rows = latest.get("rows", 0)
    label_counts = latest.get("label_counts", {})
    true_count = label_counts.get("true", 0)
    false_count = label_counts.get("false", 0)
    origin_counts = latest.get("origin_counts", {})
    top_feats = latest.get("top_features", [])

    md.append(f"version: {version}")
    md.append(f"rows: {rows} (true={true_count} | false={false_count})")

    if origin_counts:
        breakdown = ", ".join(f"{k}={v}" for k, v in origin_counts.items())
        md.append(f"by origin: {breakdown}")

    if top_feats:
        md.append(f"top features: {', '.join(top_feats)}")

    def _fmt(v):
        try:
            return "n/a" if v is None or (isinstance(v, float) and isnan(v)) else f"{v:.2f}"
        except Exception:
            return "n/a"

    metrics = latest.get("metrics", {})
    if metrics:
        md.append("metrics:")
        for model, m in metrics.items():
            md.append(
                f"{model}: ROC-AUC={_fmt(m.get('roc_auc'))} | "
                f"PR-AUC={_fmt(m.get('pr_auc'))} | "
                f"LogLoss={_fmt(m.get('logloss'))}"
            )