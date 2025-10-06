import json
from pathlib import Path

from scripts.summary_sections.common import SummaryContext
from scripts.summary_sections import influence_graph as ig

def test_influence_graph_roundtrip(tmp_path):
    models = tmp_path / "models"
    arts = tmp_path / "artifacts"
    ctx = SummaryContext(models_dir=models, artifacts_dir=arts)

    md = []
    ig.append(md, ctx)

    # JSON exists and has expected keys
    jpath = models / "influence_graph.json"
    assert jpath.exists() and jpath.stat().st_size > 0
    data = json.loads(jpath.read_text())
    assert "nodes" in data and "edges" in data and "thresholds" in data

    # CSVs exist and are non-empty
    e_csv = models / "influence_edges.csv"
    n_csv = models / "influence_nodes.csv"
    assert e_csv.exists() and e_csv.stat().st_size > 0
    assert n_csv.exists() and n_csv.stat().st_size > 0

    # PNGs exist and are non-empty
    g_png = arts / "influence_graph.png"
    b_png = arts / "influence_bar.png"
    assert g_png.exists() and g_png.stat().st_size > 0
    assert b_png.exists() and b_png.stat().st_size > 0

    # Markdown contains thresholds line
    joined = "\n".join(md)
    assert "Edges weighted by |r| × (1 − p)" in joined