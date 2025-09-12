# scripts/summary_sections/training_metadata_section.py
from __future__ import annotations
from typing import List, Optional, Dict, Any

def _fmt_metric(v) -> str:
    from math import isnan
    try:
        if v is None:
            return "n/a"
        if isinstance(v, float) and isnan(v):
            return "n/a"
        return f"{float(v):.2f}"
    except Exception:
        return "n/a"

def render(md: List[str], latest: Optional[Dict[str, Any]] = None) -> None:
    """
    Render the '📦 Latest Training Metadata' section.
    If `latest` is None, this loads via src.ml.training_metadata.load_latest_training_metadata().
    """
    md.append("\n📦 Latest Training Metadata")

    # Load if not provided
    if latest is None:
        try:
            from src.ml import training_metadata  # type: ignore
            latest = training_metadata.load_latest_training_metadata()
        except Exception as e:
            md.append(f"⚠️ Failed to load training metadata: {e}")
            latest = None

    if not latest:
        md.append("No training metadata available (yet).")
        return

    version = latest.get("version", "n/a")
    rows = latest.get("rows", 0)
    label_counts = latest.get("label_counts", {}) or {}
    true_count = label_counts.get("true", 0)
    false_count = label_counts.get("false", 0)
    origin_counts = latest.get("origin_counts", {}) or {}
    top_feats = latest.get("top_features", []) or []

    md.append(f"version: {version}")
    md.append(f"rows: {rows} (true={true_count} | false={false_count})")

    if origin_counts:
        breakdown = ", ".join(f"{k}={v}" for k, v in origin_counts.items())
        md.append(f"by origin: {breakdown}")

    if top_feats:
        md.append(f"top features: {', '.join(map(str, top_feats))}")

    metrics = latest.get("metrics", {}) or {}
    if metrics:
        md.append("metrics:")
        for model_name, m in metrics.items():
            m = m or {}
            md.append(
                f"{model_name}: ROC-AUC={_fmt_metric(m.get('roc_auc'))} | "
                f"PR-AUC={_fmt_metric(m.get('pr_auc'))} | "
                f"LogLoss={_fmt_metric(m.get('logloss'))}"
            )