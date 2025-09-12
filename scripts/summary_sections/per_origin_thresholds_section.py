# scripts/summary_sections/per_origin_thresholds_section.py
from __future__ import annotations

from typing import Dict, Any, List, Optional


def _coerce_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def _extract_p(vals: Dict[str, Any], key: str):
    """
    Try a few common places/aliases for percentiles:
      - direct keys: "p70", "p80"
      - probability aliases: "p70_proba", "p80_proba"
      - nested containers: {"percentiles": {"p70": ...}}
    """
    if vals is None:
        return None

    # direct
    if key in vals:
        f = _coerce_float(vals.get(key))
        if f is not None:
            return f

    # proba suffix
    k2 = f"{key}_proba"
    if k2 in vals:
        f = _coerce_float(vals.get(k2))
        if f is not None:
            return f

    # nested: percentiles / pct / probs
    for nest in ("percentiles", "pct", "probs", "probabilities"):
        sub = vals.get(nest)
        if isinstance(sub, dict):
            if key in sub:
                f = _coerce_float(sub.get(key))
                if f is not None:
                    return f
            if k2 in sub:
                f = _coerce_float(sub.get(k2))
                if f is not None:
                    return f

    return None


def render(md: List[str], thresholds: Optional[Dict[str, Any]] = None, max_examples: int = 2) -> None:
    """
    Render the 'Per-Origin Thresholds' section.
    If thresholds is None, loads from src.ml.thresholds.load_per_origin_thresholds().
    Shows up to `max_examples` origins that have both p70 and p80 available.
    """
    md.append("\n### 🎯 Per-Origin Thresholds")

    # Lazy import to avoid hard dependency during tests
    if thresholds is None:
        try:
            from src.ml.thresholds import load_per_origin_thresholds  # type: ignore
            thresholds = load_per_origin_thresholds() or {}
        except Exception as e:
            thresholds = {}
            md.append(f"- [warn] unable to load thresholds: {type(e).__name__}: {e}")

    if not isinstance(thresholds, dict) or not thresholds:
        md.append("- [demo] fallback thresholds in use")
        return

    shown = 0
    # stable order: origin name asc
    for origin in sorted(thresholds.keys()):
        vals = thresholds.get(origin) or {}
        p70 = _extract_p(vals, "p70")
        p80 = _extract_p(vals, "p80")

        if p70 is None or p80 is None:
            # skip rows without both values
            continue

        md.append(f"- {origin}: p70={p70:.2f}, p80={p80:.2f}")
        shown += 1
        if shown >= max_examples:
            break

    if shown == 0:
        md.append("- [demo] fallback thresholds in use")
