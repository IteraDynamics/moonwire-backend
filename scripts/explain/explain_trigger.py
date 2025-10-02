# scripts/explain/explain_trigger.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Any, Optional, Tuple
import math
import random


@dataclass
class ExplainResult:
    feature: str
    contribution: float


# ---- helpers ---------------------------------------------------------------

def _get_feature_names_from_model(model: Any) -> Optional[List[str]]:
    """
    Try to pull feature names from common sklearn conventions.
    """
    # scikit-learn >= 1.0
    names = getattr(model, "feature_names_in_", None)
    if isinstance(names, (list, tuple)):
        return list(names)
    if names is not None and hasattr(names, "tolist"):
        try:
            return list(names.tolist())
        except Exception:
            pass

    # Fallback: some pipelines stash in a custom attr
    names = getattr(model, "feature_names", None)
    if isinstance(names, (list, tuple)):
        return list(names)
    return None


def _vectorize_row(
    row: Dict[str, Any],
    feature_names: Sequence[str],
) -> List[float]:
    """
    Convert a trigger row to a dense vector aligned to feature_names.
    Missing -> 0.0, bool -> 0/1, str -> 0.0 (ignore), int/float passthrough.
    """
    vec: List[float] = []
    for f in feature_names:
        v = row.get(f, 0.0)
        if isinstance(v, bool):
            vec.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)):
            vec.append(float(v))
        else:
            # strings or unknown -> 0
            try:
                vec.append(float(v))  # maybe numeric string
            except Exception:
                vec.append(0.0)
    return vec


def _topk(items: List[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
    # Sort by abs(contribution) desc, break ties by feature name for stability
    return sorted(items, key=lambda x: (abs(x[1]), x[0]), reverse=True)[:k]


# ---- public API ------------------------------------------------------------

def explain_trigger(
    trigger_row: Dict[str, Any],
    model: Any,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Compute lightweight, model-specific per-feature contributions for a single trigger row.

    Logistic family: contribution ~= weight * value (per feature).
    Tree family:    contribution ~= feature_importance * value (approx).

    Returns a list of dicts: [{feature, contribution}] sorted by |contribution| desc.
    """
    # 1) figure out feature names
    feature_names = _get_feature_names_from_model(model)
    if not feature_names:
        # best-effort: use numeric-like keys from the row, stable-sorted
        numeric_like = []
        for k, v in trigger_row.items():
            if isinstance(v, (int, float)) or (isinstance(v, bool)):
                numeric_like.append(k)
        feature_names = sorted(numeric_like)

    x = _vectorize_row(trigger_row, feature_names)

    # 2) detect model family
    coef = getattr(model, "coef_", None)
    if coef is not None:
        # logistic / linear (coef_ shape [1, n_features] or [n_classes, n_features])
        if hasattr(coef, "tolist"):
            coef = coef.tolist()
        if isinstance(coef, list) and len(coef) > 0 and isinstance(coef[0], (list, tuple)):
            # pick first class if multi-class; we only need directional ranking
            weights = list(coef[0])
        elif isinstance(coef, (list, tuple)):
            weights = list(coef)
        else:
            # unsupported coef format -> fallback zeros
            weights = [0.0] * len(x)

        contrib = [(feature_names[i], float(weights[i]) * float(x[i])) for i in range(len(x))]
        top = _topk(contrib, top_k)
        return [{"feature": f, "contribution": c} for (f, c) in top]

    fi = getattr(model, "feature_importances_", None)
    if fi is not None:
        # tree-based approx
        if hasattr(fi, "tolist"):
            fi = fi.tolist()
        weights = list(fi) if isinstance(fi, (list, tuple)) else [0.0] * len(x)
        contrib = [(feature_names[i], float(weights[i]) * float(x[i])) for i in range(len(x))]
        top = _topk(contrib, top_k)
        return [{"feature": f, "contribution": c} for (f, c) in top]

    # 3) fallback: simple magnitude ranking of the row values (model-agnostic)
    contrib = [(feature_names[i], float(x[i])) for i in range(len(x))]
    top = _topk(contrib, top_k)
    return [{"feature": f, "contribution": c} for (f, c) in top]


# ---- demo helpers ----------------------------------------------------------

_DEMO_TOPS = {
    "reddit": ["btc_return_1h", "reddit_burst_etf", "volatility_6h"],
    "twitter": ["solana_sentiment", "volatility_6h", "news_score"],
    "rss_news": ["sec_approval_term", "btc_price_jump", "liquidity_shock"],
}

def demo_explanation_for_origin(origin: str, k: int = 3) -> List[Dict[str, Any]]:
    base = _DEMO_TOPS.get(origin, ["feature_a", "feature_b", "feature_c"])
    tops = base[:k]
    # deterministic pseudo contributions
    out = []
    for i, f in enumerate(tops):
        out.append({"feature": f, "contribution": round(0.25 - i * 0.07, 3)})
    return out
