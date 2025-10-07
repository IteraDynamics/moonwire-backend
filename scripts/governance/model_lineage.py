# scripts/governance/model_lineage.py
from __future__ import annotations

"""
Model Version Lineage & Provenance (v0.7.7)
-------------------------------------------
Builds a tolerant lineage view over versioned model directories and logs.

Public API:
    append(md: List[str], ctx: SummaryContext) -> None

Inputs (best-effort, all optional):
    - models/training_metadata.json            (global training metadata sink)
    - models/retrain_plan.json                 (latest governance retrain plan)
    - logs/trigger_history.jsonl               (append-only trigger decisions)
    - logs/label_feedback.jsonl                (append-only human labels)
    - models/v*/ (per-version subdirs; may include metrics/meta files)

Outputs:
    - models/model_lineage.json
    - artifacts/model_lineage_graph.png
    - Markdown block in CI summary

Notes:
    - Extremely tolerant to missing files/fields.
    - Demo mode: if no versioned dirs found, we emit a stable seeded lineage.
"""

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Matplotlib (Agg) for CI-safe plotting
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Optional networkx; fall back to straight matplotlib if unavailable
try:  # noqa: E402
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

# Shared helpers from summary sections
try:
    from scripts.summary_sections.common import (
        SummaryContext,
        ensure_dir,
        _read_json,
        _load_jsonl,
        _write_json,
        _iso,
        is_demo_mode,
    )
except Exception:
    # Local tolerant fallbacks if imported outside runner (tests expect main path)
    from scripts.summary_sections.common import (  # type: ignore
        SummaryContext,
        ensure_dir,
        _read_json,
        _load_jsonl,
        _write_json,
        _iso,
        is_demo_mode,
    )


# ---------------------------
# Public entrypoint
# ---------------------------

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build lineage, write JSON + PNG, and append a markdown block.
    """
    models_dir = Path(ctx.models_dir or "models")
    artifacts_dir = Path(ctx.artifacts_dir or "artifacts")
    ensure_dir(models_dir)
    ensure_dir(artifacts_dir)

    lineage = _build_lineage(ctx=ctx, models_dir=models_dir)
    _write_json(models_dir / "model_lineage.json", lineage, pretty=True)

    # Render visualization (best effort)
    try:
        _render_lineage_graph(
            lineage=lineage,
            out_png=artifacts_dir / "model_lineage_graph.png",
        )
    except Exception as e:  # keep CI summary robust
        md.append(f"\n> ⚠️ Model Lineage graph render failed: `{type(e).__name__}: {e}`\n")

    md.append(_to_markdown(lineage))


# ---------------------------
# Core lineage construction
# ---------------------------

SEMVER_RE = re.compile(r"^v(\d+)\.(\d+)(?:\.(\d+))?$", re.IGNORECASE)


@dataclass
class VersionInfo:
    version: str
    parent: Optional[str]
    source_logs: List[str]
    trigger_count: int
    label_count: int
    precision: Optional[float]
    recall: Optional[float]
    ece: Optional[float]
    brier: Optional[float]
    derived_from: str
    demo: bool = False


def _build_lineage(ctx: SummaryContext, models_dir: Path) -> Dict[str, Any]:
    """
    Assemble lineage across all versioned model directories in models_dir.
    If none exist, emit seeded demo lineage.
    """
    # Try to discover versioned directories
    version_dirs = sorted(
        [p for p in models_dir.glob("v*") if p.is_dir()],
        key=_semver_key,
    )

    training_meta = _read_json(models_dir / "training_metadata.json", default={}) or {}
    retrain_plan = _read_json(models_dir / "retrain_plan.json", default={}) or {}

    trigger_log = _load_jsonl(Path(ctx.logs_dir or "logs") / "trigger_history.jsonl")
    label_log = _load_jsonl(Path(ctx.logs_dir or "logs") / "label_feedback.jsonl")

    # If nothing discovered, seed demo
    if not version_dirs:
        return _seed_demo_lineage()

    # Build per-version entries
    versions: List[VersionInfo] = []
    prev: Optional[str] = None
    for vdir in version_dirs:
        ver = vdir.name

        # Infer parent heuristically:
        # 1) If a file "parent.txt" or meta field "parent" exists, use it.
        # 2) Else chain to previous discovered version (semver order).
        parent = _read_parent(vdir) or prev

        # Extract metrics (best-effort) from common filenames
        precision, recall, ece, brier = _read_metrics_for_version(vdir)

        # Counts from logs (best-effort matching "model_version" or "version")
        trig_cnt = _count_for_version(trigger_log, ver, keys=("model_version", "version"))
        lbl_cnt = _count_for_version(label_log, ver, keys=("model_version", "version"))

        # provenance type: try to infer from retrain plan, else "retrain"/"initial"
        derived_from = _infer_derivation(ver, parent, retrain_plan, training_meta)

        # Source logs — if we ever wrote per-version joined training logs, include them
        maybe_join = models_dir / f"training_data_{ver}.jsonl"
        src_logs = [str(maybe_join)] if maybe_join.exists() else []

        versions.append(
            VersionInfo(
                version=ver,
                parent=parent,
                source_logs=src_logs,
                trigger_count=trig_cnt,
                label_count=lbl_cnt,
                precision=precision,
                recall=recall,
                ece=ece,
                brier=brier,
                derived_from=derived_from,
                demo=False,
            )
        )
        prev = ver

    lineage = {
        "generated_at": _iso_now(),
        "versions": [vi.__dict__ for vi in versions],
        "demo": False,
    }
    return lineage


def _read_parent(vdir: Path) -> Optional[str]:
    # Try text marker
    ptxt = vdir / "parent.txt"
    if ptxt.exists():
        val = ptxt.read_text(encoding="utf-8").strip()
        return val or None
    # Try meta JSON
    for cand in ("meta.json", "training_meta.json", "metrics.json"):
        p = vdir / cand
        if p.exists():
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                for k in ("parent", "derived_from_version", "base_version"):
                    if isinstance(j.get(k), str):
                        return j[k]
            except Exception:
                pass
    return None


def _read_metrics_for_version(vdir: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Try common files/keys to extract precision/recall/ece/brier.
    """
    candidates = [
        vdir / "metrics.json",
        vdir / "eval.json",
        vdir / "meta.json",
        vdir / "training_meta.json",
    ]
    precision = recall = ece = brier = None
    for p in candidates:
        if not p.exists():
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        precision = precision if precision is not None else _get_float(j, ("precision", "P", "prec"))
        recall = recall if recall is not None else _get_float(j, ("recall", "R", "rec"))
        ece = ece if ece is not None else _get_float(j, ("ece", "calibration_ece"))
        brier = brier if brier is not None else _get_float(j, ("brier", "brier_score"))
    return precision, recall, ece, brier


def _get_float(j: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for k in keys:
        v = j.get(k)
        try:
            if v is None:
                continue
            return float(v)
        except Exception:
            continue
    return None


def _count_for_version(rows: List[Dict[str, Any]], version: str, keys: Iterable[str]) -> int:
    if not rows:
        return 0
    count = 0
    for r in rows:
        for k in keys:
            v = r.get(k)
            if isinstance(v, str) and v.strip() == version:
                count += 1
                break
    return count


def _infer_derivation(ver: str, parent: Optional[str], retrain_plan: Dict[str, Any], training_meta: Dict[str, Any]) -> str:
    """
    Heuristically infer how this version came to be.
    """
    if parent is None:
        return "initial_training"

    # Look into retrain plan candidates
    try:
        cands = list(retrain_plan.get("candidates", []))
        for c in cands:
            tgt = str(c.get("target_version") or c.get("version") or "")
            if tgt == ver:
                return "retrain"
    except Exception:
        pass

    # Check if thresholds were auto-applied
    try:
        if "threshold" in (training_meta.get("last_action", "") or "").lower():
            return "threshold_auto_apply"
    except Exception:
        pass

    # Fallback
    return "retrain"


def _iso_now() -> str:
    # local helper to avoid importing datetime everywhere
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _semver_key(p: Path) -> Tuple[int, int, int]:
    m = SEMVER_RE.match(p.name)
    if not m:
        return (math.inf, math.inf, math.inf)
    major = int(m.group(1))
    minor = int(m.group(2))
    patch = int(m.group(3) or 0)
    return (major, minor, patch)


# ---------------------------
# Visualization
# ---------------------------

def _render_lineage_graph(lineage: Dict[str, Any], out_png: Path) -> None:
    ensure_dir(out_png.parent)
    vers = lineage.get("versions", [])
    if not vers:
        # Nothing to render
        return

    # Build edges + nodes
    edges: List[Tuple[str, str, float]] = []
    node_sizes: Dict[str, float] = {}
    prec_map: Dict[str, Optional[float]] = {}
    for v in vers:
        vname = v.get("version", "")
        node_sizes[vname] = float(v.get("label_count") or v.get("trigger_count") or 10)
        prec_map[vname] = v.get("precision")

    for v in vers:
        child = v.get("version", "")
        parent = v.get("parent")
        if not child or not parent:
            continue
        p_child = prec_map.get(child)
        p_parent = prec_map.get(parent)
        delta = None
        if p_child is not None and p_parent is not None:
            delta = float(p_child) - float(p_parent)
        else:
            delta = 0.0  # unknown delta -> neutral color
        edges.append((parent, child, float(delta)))

    # Draw
    plt.figure(figsize=(7.5, 5.0), dpi=120)

    if nx is not None:
        G = nx.DiGraph()
        for n, sz in node_sizes.items():
            # scale node size; clamp minimum
            G.add_node(n, size=max(200.0, sz * 6.0))
        for u, v, w in edges:
            G.add_edge(u, v, delta=w)

        # Use a deterministic layout for stable CI plots
        pos = nx.spring_layout(G, seed=42, k=0.6)

        # Nodes
        n_sizes = [G.nodes[n]["size"] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=n_sizes, node_color="#d9e3f0", edgecolors="#6b7a99", linewidths=0.8)
        nx.draw_networkx_labels(G, pos, font_size=8)

        # Edges with colormap by delta (coolwarm)
        if edges:
            deltas = [G.edges[e]["delta"] for e in G.edges()]
            ec = nx.draw_networkx_edges(
                G,
                pos,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=12,
                edge_color=deltas,
                edge_cmap=plt.cm.coolwarm,
                width=1.8,
                connectionstyle="arc3,rad=0.06",
            )
            if ec is not None:
                cbar = plt.colorbar(ec, shrink=0.7, pad=0.02)
                cbar.set_label("Δ Precision (child - parent)", fontsize=8)
                cbar.ax.tick_params(labelsize=7)

        plt.title("Model Version Lineage (node size ≈ labels; edge color = ΔPrecision)", fontsize=10)
        plt.axis("off")
    else:
        # Minimal fallback: list as text if networkx is missing
        y = 0.9
        plt.text(0.01, 0.95, "Model Version Lineage (fallback)", fontsize=10)
        for u, v, w in edges:
            plt.text(0.05, y, f"{u} → {v} (ΔP={w:+.02f})", fontsize=8)
            y -= 0.05
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ---------------------------
# Markdown
# ---------------------------

def _to_markdown(lineage: Dict[str, Any]) -> str:
    lines = []
    lines.append("🧬 **Model Lineage & Provenance**")

    vers = lineage.get("versions", [])
    # Build quick lookup of precision per version
    prec = {v.get("version"): v.get("precision") for v in vers if v.get("version")}
    # Emit in semver order using the same ordering key
    ordered = sorted(vers, key=lambda x: _semver_key(Path(str(x.get('version', '')))))

    for v in ordered:
        child = v.get("version")
        parent = v.get("parent")
        if not child or not parent:
            continue
        p_child = prec.get(child)
        p_parent = prec.get(parent)
        d = None
        if (p_child is not None) and (p_parent is not None):
            d = float(p_child) - float(p_parent)
        deriv = v.get("derived_from", "retrain")
        if d is None:
            lines.append(f"{parent} → {child} (ΔPrecision n/a, {deriv})")
        else:
            lines.append(f"{parent} → {child} (ΔPrecision {d:+.02f}, {deriv})")

    if not any("→" in ln for ln in lines):
        lines.append("_No lineage edges discovered._")

    return "\n".join(lines) + "\n"


# ---------------------------
# Demo seeding
# ---------------------------

def _seed_demo_lineage() -> Dict[str, Any]:
    """
    Stable demo lineage used when no versioned model dirs exist.
    """
    demo_versions = [
        {
            "version": "v0.7.0",
            "parent": None,
            "source_logs": [],
            "trigger_count": 320,
            "label_count": 280,
            "precision": 0.75,
            "recall": 0.70,
            "ece": 0.05,
            "brier": 0.18,
            "derived_from": "initial_training",
            "demo": True,
        },
        {
            "version": "v0.7.1",
            "parent": "v0.7.0",
            "source_logs": ["models/training_data_v0.7.0.jsonl"],
            "trigger_count": 340,
            "label_count": 300,
            "precision": 0.78,
            "recall": 0.72,
            "ece": 0.05,
            "brier": 0.17,
            "derived_from": "retrain",
            "demo": True,
        },
        {
            "version": "v0.7.2",
            "parent": "v0.7.1",
            "source_logs": ["models/training_data_v0.7.1.jsonl"],
            "trigger_count": 355,
            "label_count": 310,
            "precision": 0.80,
            "recall": 0.74,
            "ece": 0.04,
            "brier": 0.16,
            "derived_from": "threshold_auto_apply",
            "demo": True,
        },
        {
            "version": "v0.7.5",
            "parent": "v0.7.2",
            "source_logs": ["models/training_data_v0.7.2.jsonl"],
            "trigger_count": 372,
            "label_count": 322,
            "precision": 0.84,
            "recall": 0.76,
            "ece": 0.04,
            "brier": 0.15,
            "derived_from": "drift_response",
            "demo": True,
        },
    ]
    return {
        "generated_at": _iso_now(),
        "versions": demo_versions,
        "demo": True,
    }
