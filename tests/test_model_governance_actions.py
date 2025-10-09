from pathlib import Path
import json
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
    kinds = {a["action"] for a in acts}
    assert {"promote", "rollback", "observe"} <= kinds

    # Plot exists
    assert (arts / "model_governance_actions.png").exists()

    # CI block appended
    assert any(line.startswith("🧭 Model Governance Actions") for line in md)
    assert "mode:" in md[-1]
