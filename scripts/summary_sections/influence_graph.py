# scripts/summary_sections/influence_graph.py
# v0.7.6 — Multi-Origin Influence Graph
# Converts lead/lag results (models/leadlag_analysis.json) into a directed influence graph,
# writes JSON + PNG artifacts, and appends a CI-friendly markdown block.

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Local helpers from the summary sections package
from .common import (
    SummaryContext,
    ensure_dir,
    _read_json,
    _write_json,
)

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------
# Config / simple utils
# ---------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _edge_weight(r: float, p: float) -> float:
    """Edge weight = |r| × (1 − p), clamped at 0."""
    return max(0.0, abs(float(r)) * (1.0 - float(p)))


def _derive_edges(pairs: List[Dict[str, Any]], r_min: float, p_sig: float) -> List[Dict[str, Any]]:
    """
    Build directed edges from lead/lag pairs.
      - Keep only where significant == true and |r| >= r_min and p < p_sig
      - Direction: lag > 0 => a -> b (a leads); lag < 0 => b -> a
      - Ignore lag == 0 (synchronous)
    """
    edges: List[Dict[str, Any]] = []
    for pr in pairs or []:
        try:
            if not pr.get("significant"):
                continue
            r = float(pr["r"])
            p = float(pr.get("p_value", pr.get("p", 1.0)))
            lag = float(pr.get("lag_hours", 0.0))
            a = pr.get("a") or pr.get("from") or pr.get("src")
            b = pr.get("b") or pr.get("to") or pr.get("dst")
            if a is None or b is None:
                continue
            if abs(r) < r_min or p >= p_sig:
                continue
            if lag == 0.0:
                continue
            src, dst = (a, b) if lag > 0 else (b, a)
            edges.append({"from": src, "to": dst, "r": r, "p": p, "w": _edge_weight(r, p)})
        except Exception:
            # Skip malformed rows defensively
            continue
    return edges


def _l1_normalize(values_by_key: Dict[str, float]) -> Dict[str, float]:
    """Normalize so the values sum to ~1 (keeps tests happy). All non-negative."""
    if not values_by_key:
        return {}
    s = sum(max(0.0, float(v)) for v in values_by_key.values())
    if s <= 0.0:
        return {k: 0.0 for k in values_by_key}
    return {k: max(0.0, float(v)) / s for k, v in values_by_key.items()}


def _compute_scores(edges: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Influence = weighted out-degree sum; Sensitivity = weighted in-degree sum.
    Both L1-normalized across nodes.
    """
    out_w: Dict[str, float] = {}
    in_w: Dict[str, float] = {}
    nodes = set()

    for e in edges:
        u, v, w = e["from"], e["to"], float(e["w"])
        nodes.update((u, v))
        out_w[u] = out_w.get(u, 0.0) + w
        in_w[v] = in_w.get(v, 0.0) + w

    for n in
