# scripts/governance/model_lineage.py
from __future__ import annotations

"""
Model Version Lineage & Provenance (v0.7.7)

Public API:
    append(md: List[str], ctx: SummaryContext) -> None

Outputs:
    - models/model_lineage.json
    - artifacts/model_lineage_graph.png
    - Markdown block in CI summary

Behavior:
    - Parse models/v*/ for version nodes, optional parent.txt and metrics.json
    - Build lineage edges parent -> child
    - Compute precision deltas (and carry other metrics if present)
    - If no real versions found, ALWAYS seed a 3–4 version demo lineage
      (tests expect this behavior).
    - Draw a simple lineage graph (NetworkX if available; fallback to plain matplotlib)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import os
import re
from pathlib import Path
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    import networkx as nx  # optional
except Exception:  # pragma: no cover
    nx = None  # type: ignore

from scripts.summary_sections.common import (
    SummaryContext,
    ensure_dir,
    _write_json,
    _read_json,
    _iso,
)

VERSION_DIR_RE = re.compile(r"^v\d+\.\d+(\.\d+)?$")


@dataclass
class VersionNode:
    version: str
    parent: Optional[str] = None
    trigger_count: Optional[int] = None
    label_count: Optional[int] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    ece: Optional[float] = None
    brier: Optional[float] = None
    derived_from: Optional[str] = None
    # internal (for plotting)
    _size: float = 1.0


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_metrics(mdir: Path) -> Dict[str, Any]:
    """
    Try a few common metric filenames; return {} when missing.
    """
    for fname in ("metrics.json", "eval.json", "meta.json"):
        f = mdir / fname
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def _safe_float(d: Dict[str, Any], k: str) -> Optional[float]:
    v = d.get(k)
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _discover_versions(models_dir: Path) -> Dict[str, VersionNode]:
    """
    Scan models_dir for version folders and parse:
      - parent from parent.txt or metrics/meta
      - metrics for precision/recall/f1/ece/brier and counts when available
    """
    out: Dict[str, VersionNode] = {}
    if not models_dir.exists():
        return out

    for child in models_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.strip()
        if not VERSION_DIR_RE.match(name):
            continue

        parent = None
        ptxt = child / "parent.txt"
        if ptxt.exists():
            parent = ptxt.read_text(encoding="utf-8").strip() or None

        metrics = _read_metrics(child)
        if not parent:
            parent = metrics.get("parent") or metrics.get("derived_from_version")

        node = VersionNode(
            version=name,
            parent=parent if isinstance(parent, str) and parent else None,
            precision=_safe_float(metrics, "precision"),
            recall=_safe_float(metrics, "recall"),
            f1=_safe_float(metrics, "f1") or _safe_float(metrics, "F1"),
            ece=_safe_float(metrics, "ece"),
            brier=_safe_float(metrics, "brier"),
            trigger_count=int(metrics.get("trigger_count", 0)) if str(metrics.get("trigger_count", "")).isdigit() else None,
            label_count=int(metrics.get("label_count", 0)) if str(metrics.get("label_count", "")).isdigit() else None,
            derived_from=metrics.get("derived_from") or metrics.get("action") or None,
        )
        # Node size heuristic
        if node.label_count and node.label_count > 0:
            node._size = max(1.0, math.sqrt(float(node.label_count)))
        elif node.trigger_count and node.trigger_count > 0:
            node._size = max(1.0, math.sqrt(float(node.trigger_count) * 0.5))
        else:
            node._size = 1.0

        out[name] = node

    return out


def _demo_seed() -> Dict[str, VersionNode]:
    """
    Seed a small, deterministic lineage.
    """
    # v0.7.0 -> v0.7.1 -> v0.7.2 -> v0.7.5
    seed = {
        "v0.7.0": VersionNode("v0.7.0", parent=None, precision=0.75, recall=0.70, f1=0.72, ece=0.06, brier=0.20,
                              trigger_count=300, label_count=250, derived_from="initial"),
        "v0.7.1": VersionNode("v0.7.1", parent="v0.7.0", precision=0.78, recall=0.72, f1=0.75, ece=0.055, brier=0.19,
                              trigger_count=320, label_count=270, derived_from="retrain"),
        "v0.7.2": VersionNode("v0.7.2", parent="v0.7.1", precision=0.80, recall=0.73, f1=0.76, ece=0.050, brier=0.185,
                              trigger_count=340, label_count=290, derived_from="threshold_auto_apply"),
        "v0.7.5": VersionNode("v0.7.5", parent="v0.7.2", precision=0.82, recall=0.74, f1=0.77, ece=0.048, brier=0.180,
                              trigger_count=360, label_count=305, derived_from="drift_response"),
    }
    # sizes
    for n in seed.values():
        if n.label_count:
            n._size = max(1.0, math.sqrt(float(n.label_count)))
        else:
            n._size = 1.0
    return seed


def _edges_from_nodes(nodes: Dict[str, VersionNode]) -> List[Tuple[str, str]]:
    edges = []
    for v in nodes.values():
        if v.parent and v.parent in nodes:
            edges.append((v.parent, v.version))
    return edges


def _precision_delta(nodes: Dict[str, VersionNode], parent: str, child: str) -> Optional[float]:
    p = nodes.get(parent)
    c = nodes.get(child)
    if not p or not c:
        return None
    if p.precision is None or c.precision is None:
        return None
    return c.precision - p.precision


def _write_json_artifact(models_dir: Path, nodes: Dict[str, VersionNode], demo: bool) -> Path:
    payload = {
        "generated_at": _now_utc_iso(),
        "versions": [
            {
                "version": n.version,
                "parent": n.parent,
                "source_logs": [],  # optional; fill upstream when available
                "trigger_count": n.trigger_count,
                "label_count": n.label_count,
                "precision": n.precision,
                "recall": n.recall,
                "ece": n.ece,
                "brier": n.brier,
                "derived_from": n.derived_from,
                "demo": demo,
            }
            for n in sorted(nodes.values(), key=lambda x: x.version)
        ],
        "demo": demo,
    }
    out = models_dir / "model_lineage.json"
    _write_json(out, payload, pretty=True)
    return out


def _draw_graph(artifacts_dir: Path, nodes: Dict[str, VersionNode]) -> Path:
    out = artifacts_dir / "model_lineage_graph.png"
    ensure_dir(artifacts_dir)

    # Build minimal edge attributes for coloring by ΔPrecision
    edges = _edges_from_nodes(nodes)
    edge_colors = []
    for u, v in edges:
        d = _precision_delta(nodes, u, v)
        edge_colors.append(0.0 if d is None else d)

    if nx is not None and len(nodes) <= 64:
        G = nx.DiGraph()
        for n in nodes.values():
            G.add_node(n.version, size=n._size)
        for (u, v), col in zip(edges, edge_colors):
            G.add_edge(u, v, delta=col)

        pos = nx.spring_layout(G, seed=42)
        sizes = [max(300.0, nodes[n]._size * 20.0) for n in G.nodes()]
        ec = [max(-0.1, min(0.1, G[u][v].get("delta", 0.0))) for u, v in G.edges()]
        cmap_vals = [0.5 + (d * 2.5) for d in ec]  # squeeze into [~0.25, ~0.75]
        nx.draw_networkx_nodes(G, pos, node_size=sizes)
        nx.draw_networkx_labels(G, pos, font_size=8)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", width=1.5,
                               edge_color=cmap_vals, edge_cmap=plt.cm.RdYlGn)
        plt.title("Model Lineage (parent → child), edge color = ΔPrecision")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        return out

    # Fallback: simple matplotlib layout along x-axis
    ordered = sorted(nodes.values(), key=lambda n: n.version)
    x = list(range(len(ordered)))
    y = [0.0 for _ in ordered]

    plt.figure(figsize=(max(6, len(ordered) * 1.2), 2.6))
    # Draw nodes
    for i, n in enumerate(ordered):
        plt.scatter([x[i]], [y[i]], s=max(80.0, n._size * 10.0))
        plt.text(x[i], y[i] + 0.05, n.version, ha="center", va="bottom",
                 fontsize=8, rotation=0)

    # Draw edges with color by ΔPrecision
    for u, v in edges:
        iu = next((i for i, n in enumerate(ordered) if n.version == u), None)
        iv = next((i for i, n in enumerate(ordered) if n.version == v), None)
        if iu is None or iv is None:
            continue
        d = _precision_delta(nodes, u, v) or 0.0
        color = "green" if d > 0 else "red"
        plt.annotate(
            "",
            xy=(iv, y[iv]),
            xytext=(iu, y[iu]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

    plt.title("Model Lineage (parent → child), edge color = ΔPrecision")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def _append_markdown(md: List[str], nodes: Dict[str, VersionNode], demo: bool) -> None:
    md.append("🧬 Model Lineage & Provenance")
    edges = _edges_from_nodes(nodes)
    if not edges:
        md.append("No lineage edges discovered.\n")
        return

    lines = []
    for u, v in sorted(edges, key=lambda e: e[0]):
        d = _precision_delta(nodes, u, v)
        delta_txt = "unknown"
        if d is not None:
            sign = "+" if d >= 0 else ""
            delta_txt = f"{sign}{d:.2f}"
        action = nodes[v].derived_from or "retrain"
        lines.append(f"{u} → {v} (ΔPrecision {delta_txt}, {action})")
    md.append("\n".join(lines))
    md.append("")  # spacing


def _compute_lineage(ctx: SummaryContext) -> Tuple[Dict[str, VersionNode], bool]:
    """
    Return (nodes, demo_used)

    Tests require: when no real versions exist, we MUST seed demo lineage.
    """
    models_dir = ctx.models_dir
    ensure_dir(models_dir)
    nodes = _discover_versions(models_dir)
    if nodes:
        return nodes, False

    # Always seed when nothing is discovered (demo_used=True).
    return _demo_seed(), True


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Orchestrate lineage build + artifacts + markdown append.
    Safe to call in CI even with missing inputs.
    """
    try:
        nodes, demo_used = _compute_lineage(ctx)

        # Always emit JSON so artifact step is predictable
        jpath = _write_json_artifact(ctx.models_dir, nodes, demo_used)

        # Draw graph (works fine for demo as well)
        _ = _draw_graph(ctx.artifacts_dir, nodes if nodes else {})

        # Append markdown
        _append_markdown(md, nodes, demo_used)

    except Exception as e:
        md.append(f"🧬 Model Lineage & Provenance failed: {type(e).__name__}: {e}\n")
