# scripts/summary_sections/influence_graph.py
# v0.7.6 — Multi-Origin Influence Graph
# Converts lead/lag results (models/leadlag_analysis.json) into a directed influence graph,
# writes JSON + PNG artifacts, and appends a CI-friendly markdown block.

from __future__ import annotations

import math
import os
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

    for n in nodes:
        out_w.setdefault(n, 0.0)
        in_w.setdefault(n, 0.0)

    return _l1_normalize(out_w), _l1_normalize(in_w)


# ---------------------------
# Plotting (imports deferred to avoid import-time failures)
# ---------------------------

def _plot_graph(edges: List[Dict[str, Any]], out_png: Path) -> None:
    """Directed graph; uses networkx if available, else a circular fallback."""
    ensure_dir(out_png.parent)

    # Defer heavy imports
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa

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
        # Fall through to manual plotting below
        pass

    # Fallback: simple circular layout with arrow annotations
    nodes = sorted({n for e in edges for n in (e["from"], e["to"])})
    n = max(1, len(nodes))
    coords = {
        nodes[i]: (
            math.cos(2 * math.pi * i / n),
            math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa

    plt.figure(figsize=(6, 5), dpi=160)
    for name, (x, y) in coords.items():
        plt.scatter([x], [y], s=400)
        plt.text(x, y, name, ha="center", va="center", fontsize=10)
    for e in edges:
        x1, y1 = coords[e["from"]]
        x2, y2 = coords[e["to"]]
        lw = 1.0 + 6.0 * float(e["w"])
        plt.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=lw),
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        plt.text(mx, my, f"r={e['r']:.2f} p={e['p']:.2f}", fontsize=8, ha="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _plot_bar(influence: Dict[str, float], sensitivity: Dict[str, float], out_png: Path) -> None:
    """Grouped bar chart: Influence vs Sensitivity per origin."""
    ensure_dir(out_png.parent)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa
    import numpy as np  # noqa

    keys = sorted(set(influence) | set(sensitivity))
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

    # Deterministic demo for empty/missing input
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

    # thresholds
    r_min = _get_env_float("MW_INFLUENCE_MIN_R", 0.30)
    p_sig = _get_env_float("MW_INFLUENCE_MIN_SIG", 0.05)

    pairs, demo = _load_pairs(models_dir)
    edges = _derive_edges(pairs, r_min=r_min, p_sig=p_sig)
    influence, sensitivity = _compute_scores(edges)

    # JSON artifact
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

    # PNG artifacts
    _plot_graph(edges, artifacts_dir / "influence_graph.png")
    _plot_bar(influence, sensitivity, artifacts_dir / "influence_bar.png")

    # Markdown block
    md.append("")
    md.append("🌐 **Multi-Origin Influence Graph (72 h)**")
    if edges:
        for e in edges:
            md.append(f"{e['from']} → {e['to']} (r={e['r']:.2f} p={e['p']:.2f})  ")
    else:
        md.append("_No significant directional edges under current thresholds._  ")

    if nodes_list:
        ranked = sorted(nodes_list, key=lambda k: influence.get(k, 0.0), reverse=True)
        parts = [f"{k} {influence.get(k, 0.0):.2f}" for k in ranked]
        md.append("Influence scores: " + " | ".join(parts))

    md.append("")
    md.append("_Edges weighted by |r| × (1 − p). p<0.05 = significant. Scores L1-normalized (sum≈1)._")
