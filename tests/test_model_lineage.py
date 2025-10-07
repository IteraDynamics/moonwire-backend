# tests/test_model_lineage.py
from __future__ import annotations

import json
from pathlib import Path

from scripts.summary_sections.common import SummaryContext, ensure_dir
from scripts.governance import model_lineage


def _read_json(path: Path):
    return json.loads(path.read_text())


def test_demo_seed_when_no_versions(tmp_path: Path):
    models = tmp_path / "models"
    arts = tmp_path / "artifacts"
    ensure_dir(models); ensure_dir(arts)
    ctx = SummaryContext(models_dir=models, artifacts_dir=arts)

    md: list[str] = []
    model_lineage.append(md, ctx)

    # JSON emitted
    jpath = models / "model_lineage.json"
    assert jpath.exists()
    j = _read_json(jpath)
    assert "generated_at" in j
    assert isinstance(j.get("versions"), list) and len(j["versions"]) >= 3
    assert j.get("demo") is True

    # PNG emitted
    ppath = arts / "model_lineage_graph.png"
    assert ppath.exists()
    assert ppath.stat().st_size > 0

    # Markdown contains header and at least two edges
    joined = "\n".join(md)
    assert "Model Lineage & Provenance" in joined
    assert "→" in joined


def test_real_versions_order_and_edges(tmp_path: Path):
    models = tmp_path / "models"
    arts = tmp_path / "artifacts"
    ensure_dir(models); ensure_dir(arts)

    # Create simple real version dirs with minimal metrics
    v070 = models / "v0.7.0"; v071 = models / "v0.7.1"; v072 = models / "v0.7.2"
    for d in (v070, v071, v072):
        ensure_dir(d)

    # Parent chain via parent.txt
    (v071 / "parent.txt").write_text("v0.7.0")
    (v072 / "parent.txt").write_text("v0.7.1")

    # Minimal metrics to drive deltas
    (v070 / "metrics.json").write_text(json.dumps({"precision": 0.75}))
    (v071 / "metrics.json").write_text(json.dumps({"precision": 0.78}))
    (v072 / "metrics.json").write_text(json.dumps({"precision": 0.80}))

    ctx = SummaryContext(models_dir=models, artifacts_dir=arts)
    md: list[str] = []
    model_lineage.append(md, ctx)

    j = _read_json(models / "model_lineage.json")
    versions = {v["version"]: v for v in j["versions"]}
    assert versions["v0.7.1"]["parent"] == "v0.7.0"
    assert versions["v0.7.2"]["parent"] == "v0.7.1"

    # Check markdown deltas
    joined = "\n".join(md)
    assert "v0.7.0 → v0.7.1 (ΔPrecision +0.03" in joined
    assert "v0.7.1 → v0.7.2 (ΔPrecision +0.02" in joined

    # PNG exists
    graph = arts / "model_lineage_graph.png"
    assert graph.exists() and graph.stat().st_size > 0
