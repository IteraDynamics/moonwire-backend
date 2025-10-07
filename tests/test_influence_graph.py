# tests/test_influence_graph.py

import json
import math
import os
from pathlib import Path

from scripts.summary_sections import influence_graph as ig
from scripts.summary_sections.common import ensure_dir, SummaryContext


def _write(path: Path, obj):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_demo_mode_when_missing_pairs(tmp_path):
    models = tmp_path / "models"
    arts = tmp_path / "artifacts"
    ctx = SummaryContext(models_dir=models, artifacts_dir=arts)

    md = []
    ig.append(md, ctx)

    # JSON created and marked demo
    j = models / "influence_graph.json"
    assert j.exists()
    data = json.loads(j.read_text())
    assert data.get("demo") is True
    assert "nodes" in data and "edges" in data

    # PNGs exist and are non-empty
    g = arts / "influence_graph.png"
    b = arts / "influence_bar.png"
    assert g.exists() and g.stat().st_size > 0
    assert b.exists() and b.stat().st_size > 0

    # Markdown appended
    text = "\n".join(md)
    assert "Multi-Origin Influence Graph" in text


def test_edges_and_scores_from_pairs(tmp_path):
    models = tmp_path / "models"
    arts = tmp_path / "artifacts"
    ctx = SummaryContext(models_dir=models, artifacts_dir=arts)

    pairs = [
        {"a": "reddit",  "b": "twitter", "lag_hours": 2.0,  "r": 0.60, "p_value": 0.02, "significant": True},   # r->t
        {"a": "market",  "b": "reddit",  "lag_hours": -1.0, "r": 0.50, "p_value": 0.01, "significant": True},   # r->m
        {"a": "twitter", "b": "market",  "lag_hours": 0.0,  "r": 0.40, "p_value": 0.20, "significant": False},  # ignored
    ]
    _write(models / "leadlag_analysis.json", pairs)

    os.environ["MW_INFLUENCE_MIN_R"] = "0.30"
    os.environ["MW_INFLUENCE_MIN_SIG"] = "0.05"

    md = []
    ig.append(md, ctx)

    data = json.loads((models / "influence_graph.json").read_text())
    assert data["demo"] is False

    edge_set = {(e["from"], e["to"]) for e in data["edges"]}
    assert ("reddit", "twitter") in edge_set
    assert ("reddit", "market") in edge_set
    assert ("twitter", "market") not in edge_set  # synchronous/non-sig ignored

    nodes = {n["origin"]: (n["influence"], n["sensitivity"]) for n in data["nodes"]}
    for key in ["reddit", "twitter", "market"]:
        assert key in nodes

    infl_sum = sum(v[0] for v in nodes.values())
    sens_sum = sum(v[1] for v in nodes.values())
    # L1-normalized: sums ~ 1 (or 0 if no edges)
    assert math.isclose(infl_sum, 1.0, rel_tol=1e-6) or math.isclose(infl_sum, 0.0, rel_tol=1e-6)
    assert math.isclose(sens_sum, 1.0, rel_tol=1e-6) or math.isclose(sens_sum, 0.0, rel_tol=1e-6)

    # Artifacts exist
    assert (arts / "influence_graph.png").exists()
    assert (arts / "influence_bar.png").exists()

    # Markdown contains edges and summary
    md_text = "\n".join(md)
    assert "reddit → twitter" in md_text
    assert "reddit → market" in md_text
    assert "Influence scores:" in md_text
