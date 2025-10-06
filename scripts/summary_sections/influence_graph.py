# scripts/summary_sections/influence_graph.py
# v0.7.6 — Multi-Origin Influence Graph
# Robustly parses lead/lag results (models/leadlag_analysis.json) into a directed influence graph,
# writes JSON + PNG artifacts, and appends a CI-friendly markdown block.

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from .common import (
    SummaryContext,
    ensure_dir,
    _read_json,
    _write_json,
)

# ---------------------------
# Config / simple utils
# ---------------------------

_PAIR_SPLIT_RE = re.compile(r"\s*(?:-|–|—|→|->|=>)\s*")

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def _coerce_float(d: Dict[str, Any], keys: List[str], default: float) -> float:
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    return default

def _coerce_str(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            return str(d[k])
    return None

def _edge_weight(r: float, p: float) -> float:
    """Edge weight = |r| × (1 − p), clamped at 0."""
    return max(0.0, abs(float(r)) * (1.0 - float(p)))

def _is_significant(pr: Dict[str, Any], r_min: float, p_sig: float) -> bool:
    """
    Robust significance check:
      - If 'significant' key exists, require it True.
      - Always also enforce p < p_sig and |r| ≥ r_min.
      - If 'significant' is missing, rely on thresholds only.
    """
    r = _coerce_float(pr, ["r", "corr", "rho"], float("nan"))
    p = _coerce_float(pr, ["p_value", "p", "pval", "pvalue"], float("nan"))
    if not (r == r and p == p):  # NaN check
        return False
    if "significant" in pr and not bool(pr["significant"]):
        return False
    return (abs(r) >= r_min) and (p < p_sig)

def _extract_nodes(pr: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Try to extract (a, b, lag_hours) from many plausible schemas.
    Priority:
      1) explicit pairs: (a,b) / (from,to) / (src,dst) / (origin_a,origin_b) / (x,y)
      2) leader/follower (ignore lag sign)
      3) single "pair" string like "reddit–twitter", interpret lag sign to infer direction
    Returns (a, b, lag_hours) where positive lag means a leads b.
    """
    # direct two-key forms
    a = _coerce_str(pr, ["a", "from", "src", "origin_a", "x", "left"])
    b = _coerce_str(pr, ["b", "to", "dst", "origin_b", "y", "right"])
    lag = _coerce_float(pr, ["lag_hours", "lag", "lag_h"], 0.0)

    if a and b:
        # Interpret sign: lag > 0 means a leads b; lag < 0 means b leads a
        if lag < 0:
            a, b = b, a
            lag = abs(lag)
        return a, b, lag

    # leader/follower explicitly given
    leader = _coerce_str(pr, ["leader"])
    follower = _coerce_str(pr, ["follower"])
    if leader and follower:
        # Use given direction; lag magnitude if present, else 0 (we only need direction)
        lag = abs(_coerce_float(pr, ["lag_hours", "lag", "lag_h"], 0.0))
        return leader, follower, lag

    # parse "pair" strings like "reddit–twitter" or "reddit -> twitter"
    pair = _coerce_str(pr, ["pair", "pair_key", "pair_id"])
    if pair:
        parts = _PAIR_SPLIT_RE.split(pair)
        if len(parts) == 2 and parts[0] and parts[1]:
            a, b = parts[0].strip(), parts[1].strip()
            lag = _coerce_float(pr, ["lag_hours", "lag", "lag_h"], 0.0)
            if lag < 0:
                a, b = b, a
                lag = abs(lag)
            return a, b, lag

    return None, None, None

def _derive_edges(pairs: List[Dict[str, Any]], r_min: float, p_sig: float) -> List[Dict[str, Any]]:
    """
    Build directed edges from lead/lag pairs.
      - Direction: positive lag => A -> B (A leads B).
      - Keep only significant pairs per _is_significant.
      - Robust to varied input schemas.
    """
    edges: List[Dict[str, Any]] = []
    for pr in pairs or []:
        try:
            if not _is_significant(pr, r_min=r_min, p_sig=p_sig):
                continue

            r = _coerce_float(pr, ["r", "corr", "rho"], float("nan"))
            p = _coerce_float(pr, ["p_value", "p", "pval", "pvalue"], float("nan"))
            a, b, lag = _extract_nodes(pr)

            if r != r or p != p or a is None or b is None:
                continue

            # If lag not provided or zero in a/b or leader/follower forms, we already encoded direction in (a,b)
            # but we still skip truly synchronous cases if explicitly given as 0 across schemas.
            if lag is not None and float(lag) == 0.0:
                # if we got leader/follower without a meaningful lag, still treat as directional
                # unless data explicitly marks synchronous; we can't detect that here, so allow.
                pass

            edges.append({"from": a, "to": b, "r": float(r), "p": float(p), "w": _edge_weight(r, p)})
        except Exception:
            continue
    return edges

def _l1_normalize(values_by_key: Dict[str, float]) -> Dict[str, float]:
    """Normalize so the values sum to ~1. All non-negative."""
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

    for n in nodes:
        out_w.setdefault(n, 0.0)
        in_w.setdefault(n, 0.0)

    return _l1_normalize(out_w), _l1_normalize(in_w)

# ---------------------------
# Plotting (imports deferred)
# ---------------------------

def _plot_graph(edges: List[Dict[str, Any]], out_png: Path) -> None:
    """Directed graph; uses networkx if available, else a circular fallback. Handles no-edge cases."""
    ensure_dir(out_png.parent)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa

    if not edges:
        plt.figure(figsize=(6, 4), dpi=160)
        plt.text(0.5, 0.5, "No significant edges", ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return

    try:
        try:
            import networkx as nx  # optional
        except Exception:
            nx = None

        if nx is not None:
            G = nx.DiGraph()
            for e in edges:
                G.add_edge(e["from"], e["to"], weight=e["w"], r=e["r"], p=e["p"])
            pos = nx.spring_layout(G, seed=7) if len(G.nodes) > 2 else nx.circular_layout(G)
            plt.figure(figsize=(6, 5), dpi=160)
            nx.draw_networkx_nodes(G, pos, node_size=1200)
            nx.draw_networkx_labels(G, pos, font_size=10)
            widths = [1.0 + 6.0 * G[u][v]["weight"] for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, arrows=True, width=widths, arrowstyle="-|>", arrowsize=18)
            labels = {(u, v): f"r={G[u][v]['r']:.2f} p={G[u][v]['p']:.2f}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()
            return
    except Exception:
        pass

    # Manual circular layout
    nodes = sorted({n for e in edges for n in (e["from"], e["to"])})
    if not nodes:
        plt.figure(figsize=(6, 4), dpi=160)
        plt.text(0.5, 0.5, "No significant edges", ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return

    coords: Dict[str, Tuple[float, float]] = {}
    n = len(nodes)
    for i, name in enumerate(nodes):
        coords[name] = (
            math.cos(2 * math.pi * i / n),
            math.sin(2 * math.pi * i / n),
        )

    plt.figure(figsize=(6, 5), dpi=160)
    for name, (x, y) in coords.items():
        plt.scatter([x], [y], s=400)
        plt.text(x, y, name, ha="center", va="center", fontsize=10)
    for e in edges:
        x1, y1 = coords[e["from"]]
        x2, y2 = coords[e["to"]]
        lw = 1.0 + 6.0 * float(e["w"])
        plt.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=lw))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        plt.text(mx, my, f"r={e['r']:.2f} p={e['p']:.2f}", fontsize=8, ha="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def _plot_bar(influence: Dict[str, float], sensitivity: Dict[str, float], out_png: Path) -> None:
    """Grouped bar chart: Influence vs Sensitivity per origin. Handles empty data."""
    ensure_dir(out_png.parent)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa
    import numpy as np  # noqa

    keys = sorted(set(influence) | set(sensitivity))
    if not keys:
        plt.figure(figsize=(7, 4), dpi=160)
        plt.text(0.5, 0.5, "No nodes (no significant edges)", ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return

    vals_i = [float(influence.get(k, 0.0)) for k in keys]
    vals_s = [float(sensitivity.get(k, 0.0)) for k in keys]
    x = np.arange(len(keys))
    w = 0.38

    plt.figure(figsize=(7, 4), dpi=160)
    plt.bar(x - w / 2, vals_i, width=w, label="Influence")
    plt.bar(x + w / 2, vals_s, width=w, label="Sensitivity")
    plt.xticks(x, keys)
    plt.ylabel("Normalized score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ---------------------------
# I/O
# ---------------------------

def _load_pairs(models_dir: Path) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Read models/leadlag_analysis.json or return deterministic demo pairs.
    Returns (pairs, demo_flag).
    """
    jpath = Path(models_dir) / "leadlag_analysis.json"
    pairs = _read_json(jpath, default=None)
    if pairs:
        return pairs, False

    demo_pairs = [
        {"a": "reddit", "b": "twitter", "lag_hours": 1.0, "r": 0.62, "p_value": 0.03, "significant": True},
        {"a": "reddit", "b": "market",  "lag_hours": 2.0, "r": 0.42, "p_value": 0.01, "significant": True},
        {"a": "twitter","b": "market",  "lag_hours": 0.0, "r": 0.38, "p_value": 0.12, "significant": False},
    ]
    return demo_pairs, True

# ---------------------------
# Public API
# ---------------------------

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Generate the influence graph artifacts and append a markdown block.
    Artifacts:
      - models/influence_graph.json
      - artifacts/influence_graph.png
      - artifacts/influence_bar.png
    """
    models_dir = Path(ctx.models_dir or "models")
    artifacts_dir = Path(ctx.artifacts_dir or "artifacts")
    ensure_dir(models_dir)
    ensure_dir(artifacts_dir)

    r_min = _get_env_float("MW_INFLUENCE_MIN_R", 0.30)
    p_sig = _get_env_float("MW_INFLUENCE_MIN_SIG", 0.05)

    pairs, demo = _load_pairs(models_dir)
    edges = _derive_edges(pairs, r_min=r_min, p_sig=p_sig)
    influence, sensitivity = _compute_scores(edges)

    nodes_list = sorted(set(list(influence.keys()) + list(sensitivity.keys())))
    graph_json = {
        "generated_at": _utc_now_iso(),
        "nodes": [
            {"origin": k, "influence": float(influence.get(k, 0.0)), "sensitivity": float(sensitivity.get(k, 0.0))}
            for k in nodes_list
        ],
        "edges": [{"from": e["from"], "to": e["to"], "r": float(e["r"]), "p": float(e["p"])} for e in edges],
        "demo": bool(demo),
    }
    _write_json(models_dir / "influence_graph.json", graph_json, pretty=True)

    _plot_graph(edges, artifacts_dir / "influence_graph.png")
    _plot_bar(influence, sensitivity, artifacts_dir / "influence_bar.png")

    md.append("")
    md.append("🌐 **Multi-Origin Influence Graph (72 h)**")
    if edges:
        for e in edges:
            md.append(f"{e['from']} → {e['to']} (r={e['r']:.2f} p={e['p']:.2f})  ")
        # Influence summary (sorted)
        if nodes_list:
            ranked = sorted(nodes_list, key=lambda k: influence.get(k, 0.0), reverse=True)
            parts = [f"{k} {influence.get(k, 0.0):.2f}" for k in ranked]
            md.append("Influence scores: " + " | ".join(parts))
    else:
        md.append("_No significant directional edges under current thresholds._  ")

    md.append("")
    md.append("_Edges weighted by |r| × (1 − p). p<0.05 = significant. Scores L1-normalized (sum≈1)._")