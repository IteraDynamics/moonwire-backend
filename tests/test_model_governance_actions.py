# tests/test_model_governance_actions.py
from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.summary_sections.common import SummaryContext, ensure_dir
from scripts.governance import model_governance_actions as mga


def test_plan_demo_seed(tmp_path: Path, monkeypatch):
    # demo mode on
    monkeypatch.setenv("DEMO_MODE", "true")
    models = tmp_path / "models"
    arts = tmp_path / "artifacts"
    ensure_dir(models); ensure_dir(arts)

    # minimal lineage + trend to exercise path
    (models / "model_lineage.json").write_text(json.dumps({
        "generated_at": "2025-10-08T00:00:00Z",
        "versions": [{"version": "v0.7.5"}, {"version": "v0.7.6"}],
        "demo": True
    }))
    (models / "model_performance_trend.json").write_text(json.dumps({
        "generated_at": "2025-10-08T00:00:00Z",
        "window_hours": 72,
        "versions": [
            {"version": "v0.7.5", "precision_trend": "declining", "ece_trend": "worsening", "precision_delta": -0.03, "ece_delta": 0.012},
            {"version": "v0.7.6", "precision_trend": "improving", "ece_trend": "improving", "precision_delta": 0.02, "ece_delta": -0.01}
        ],
        "demo": True
    }))

    md = []
    ctx = SummaryContext(models_dir=models, artifacts_dir=arts)
    mga.append(md, ctx)

    # JSON written
    out = models / "model_governance_actions.json"
    assert out.exists()
    j = json.loads(out.read_text())
    assert j.get("mode") in ("dryrun", "apply")
    acts = j.get("actions", [])
    assert isinstance(acts, list) and len(acts) >= 3  # demo ensures ≥1 of each type

    # CI block
    assert any("Model Governance Actions" in line for line in md)

    # Plot exists (placeholder acceptable)
    p = arts / "model_governance_actions.png"
    assert p.exists()
    assert p.stat().st_size > 0


def test_live_logic(tmp_path: Path, monkeypatch):
    # live-ish (no demo seed)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("MW_DEMO", raising=False)
    monkeypatch.setenv("MW_GOV_ACTION_MIN_PRECISION", "0.75")
    monkeypatch.setenv("MW_GOV_ACTION_MAX_ECE", "0.06")

    models = tmp_path / "models"
    arts = tmp_path / "artifacts"
    ensure_dir(models); ensure_dir(arts)

    (models / "model_lineage.json").write_text(json.dumps({
        "generated_at": "2025-10-08T00:00:00Z",
        "versions": [
            {"version": "v0.7.5", "precision": 0.73, "ece": 0.07},
            {"version": "v0.7.6", "precision": 0.78, "ece": 0.05}
        ],
        "demo": False
    }))
    (models / "model_performance_trend.json").write_text(json.dumps({
        "generated_at": "2025-10-08T00:00:00Z",
        "window_hours": 72,
        "versions": [
            {"version": "v0.7.5", "precision_trend": "declining", "ece_trend": "worsening"},
            {"version": "v0.7.6", "precision_trend": "improving", "ece_trend": "improving"}
        ],
        "demo": False
    }))

    md = []
    ctx = SummaryContext(models_dir=models, artifacts_dir=arts)
    mga.append(md, ctx)

    plan = json.loads((models / "model_governance_actions.json").read_text())
    acts = {a["version"]: a for a in plan["actions"]}

    assert acts["v0.7.5"]["action"] in ("rollback", "observe")
    assert "confidence" in acts["v0.7.5"]

    assert acts["v0.7.6"]["action"] in ("promote", "observe")
    assert "confidence" in acts["v0.7.6"]

    # CI block rendered
    assert any("mode:" in line for line in md)
