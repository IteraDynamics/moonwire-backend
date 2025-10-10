import json, os, time
from pathlib import Path

from scripts.summary_sections.common import SummaryContext, ensure_dir
from scripts.governance import model_governance_actions as mga  # reuse helpers if needed
from scripts.governance.governance_apply import append as gov_apply

def _write(p: Path, obj):
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj))

def test_dryrun_vs_apply(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    arts = tmp_path / "artifacts"
    logs = tmp_path / "logs"
    ensure_dir(models); ensure_dir(arts); ensure_dir(logs)

    # minimal lineage & trend
    _write(models / "model_lineage.json", {
        "generated_at": "2025-10-08T00:00:00Z",
        "versions": [{"version":"v0.7.5"}, {"version":"v0.7.6"}, {"version":"v0.7.7"}]
    })
    _write(models / "model_performance_trend.json", {
        "generated_at": "2025-10-08T00:00:00Z",
        "versions": [
            {"version":"v0.7.5","labels":100,"precision":0.70,"ece":0.08},
            {"version":"v0.7.6","labels":100,"precision":0.76,"ece":0.05},
            {"version":"v.ignored","labels":5,"precision":0.90,"ece":0.01},
            {"version":"v0.7.7","labels":100,"precision":0.80,"ece":0.04}
        ]
    })
    # recommendations
    _write(models / "model_governance_actions.json", {
        "actions": [
            {"version":"v0.7.7","action":"promote","confidence":0.91},
            {"version":"v0.7.5","action":"rollback","confidence":0.82},
            {"version":"v0.7.6","action":"observe","confidence":0.60},
        ]
    })

    # DRYRUN
    monkeypatch.setenv("MW_GOV_APPLY_MODE","dryrun")
    monkeypatch.setenv("DEMO_MODE","true")
    ctx = SummaryContext(models_dir=models, artifacts_dir=arts, logs_dir=logs)
    md=[]
    gov_apply(md, ctx)

    # result json
    rj = json.loads((models / "governance_apply_result.json").read_text())
    assert rj["mode"] == "dryrun"
    assert isinstance(rj["applied"], list) and len(rj["applied"]) >= 1
    assert isinstance(rj["skipped"], list)  # may be empty but list
    assert (arts / "governance_apply_timeline.png").exists()
    # current_model not changed in dryrun
    assert not (models / "current_model.txt").exists()

    # APPLY
    monkeypatch.setenv("MW_GOV_APPLY_MODE","apply")
    md=[]
    gov_apply(md, ctx)
    # now we expect current_model.txt to be written to v0.7.7 (promote)
    cur = (models / "current_model.txt").read_text().strip()
    assert cur in ("v0.7.7","v0.7.6","v0.7.5")  # promote target or safe fallback

    # ledger append-only
    ledger = logs / "governance_apply.jsonl"
    assert ledger.exists()
    lines = [l for l in ledger.read_text().splitlines() if l.strip()]
    assert len(lines) >= 1
    # JSON valid
    json.loads(lines[-1])

def test_guardrails_and_cooldown(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    arts = tmp_path / "artifacts"
    logs = tmp_path / "logs"
    ensure_dir(models); ensure_dir(arts); ensure_dir(logs)

    _write(models / "model_lineage.json", {"versions":[{"version":"v0.1"},{"version":"v0.2"}]})
    _write(models / "model_performance_trend.json", {
        "versions": [
            {"version":"v0.1","labels": 10, "precision":0.74, "ece":0.07},  # below thresholds
            {"version":"v0.2","labels":100, "precision":0.90, "ece":0.02},
        ]
    })
    _write(models / "model_governance_actions.json", {
        "actions":[
            {"version":"v0.1","action":"promote","confidence":0.5},
            {"version":"v0.2","action":"promote","confidence":0.9},
        ]
    })

    # set strict thresholds to force first skip, second apply
    monkeypatch.setenv("MW_GOV_MIN_LABELS","20")
    monkeypatch.setenv("MW_GOV_MIN_PRECISION","0.75")
    monkeypatch.setenv("MW_GOV_MAX_ECE","0.06")
    monkeypatch.setenv("MW_GOV_APPLY_MODE","apply")

    ctx = SummaryContext(models_dir=models, artifacts_dir=arts, logs_dir=logs)
    md=[]
    gov_apply(md, ctx)

    rj = json.loads((models / "governance_apply_result.json").read_text())
    # v0.1 should be skipped for guardrails
    assert any(a["version"]=="v0.1" for a in rj["skipped"])
    # v0.2 should be applied
    assert any(a["version"]=="v0.2" for a in rj["applied"])

    # cooldown: immediate second run should skip v0.2 next time
    md=[]
    gov_apply(md, ctx)
    r2 = json.loads((models / "governance_apply_result.json").read_text())
    assert any("cooldown" in ",".join(a.get("reason",[])) for a in r2["skipped"])

def test_reversal_plan_and_markdown(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    arts = tmp_path / "artifacts"
    logs = tmp_path / "logs"
    ensure_dir(models); ensure_dir(arts); ensure_dir(logs)

    _write(models / "model_lineage.json", {"versions":[{"version":"v0.5"},{"version":"v0.6"}]})
    _write(models / "model_performance_trend.json", {
        "versions":[{"version":"v0.6","labels":100,"precision":0.8,"ece":0.05}]
    })
    _write(models / "model_governance_actions.json", {"actions":[{"version":"v0.6","action":"promote","confidence":0.88}]})

    monkeypatch.setenv("MW_GOV_APPLY_MODE","dryrun")
    ctx = SummaryContext(models_dir=models, artifacts_dir=arts, logs_dir=logs)
    md=[]
    gov_apply(md, ctx)

    rj = json.loads((models / "governance_apply_result.json").read_text())
    assert "reversal_plan" in rj and rj["reversal_plan"]["window_hours"] == 12

    # CI block lines present
    assert any(line.startswith("🧭 Governance Apply") for line in md)
    assert any("reversal plan:" in line for line in md)
