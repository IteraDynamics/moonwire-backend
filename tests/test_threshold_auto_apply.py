# tests/test_threshold_auto_apply.py
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import importlib

from scripts.summary_sections.common import SummaryContext, _iso


def _write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_auto_apply_applies_and_skips(monkeypatch, tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)

    # Guard knobs
    monkeypatch.setenv("MW_THR_MIN_PRECISION", "0.75")
    monkeypatch.setenv("MW_THR_MIN_LABELS", "10")
    monkeypatch.setenv("MW_THR_MAX_DELTA", "0.10")
    monkeypatch.setenv("MW_THR_ALLOW_LARGE_JUMP", "false")

    # Current thresholds
    _write(models / "per_origin_thresholds.json", {"reddit": 0.50, "twitter": 0.50})

    # Recommendations
    reco = {
        "window_hours": 72,
        "objective": {"type": "precision_min_recall_max", "min_precision": 0.75},
        "per_origin": [
            {"origin": "reddit", "current": 0.50, "recommended": 0.56, "delta": 0.06},
            {"origin": "twitter", "current": 0.50, "recommended": 0.47, "delta": -0.03},
            {"origin": "rss_news", "current": 0.50, "recommended": 0.62, "delta": 0.12},  # will be large-jump skip
        ],
    }
    _write(models / "threshold_recommendations.json", reco)

    # Backtest artifact with "after" metrics + risk flags
    bt = {
        "window_hours": 72,
        "objective": {"type": "precision_min_recall_max", "min_precision": 0.75},
        "per_origin": [
            {
                "origin": "reddit",
                "current": {"thr": 0.50, "precision": 0.75, "recall": 0.54, "f1": 0.63, "labels": 20, "triggers": 10},
                "recommended": {"thr": 0.56, "precision": 0.78, "recall": 0.62, "f1": 0.69, "labels": 20, "triggers": 13},
                "risk": ["ok"],
            },
            {
                "origin": "twitter",
                "current": {"thr": 0.50, "precision": 0.75, "recall": 0.50, "f1": 0.60, "labels": 12, "triggers": 6},
                "recommended": {"thr": 0.47, "precision": 0.76, "recall": 0.56, "f1": 0.65, "labels": 12, "triggers": 8},
                "risk": ["precision-drop-risk"],  # force skip by risk flag
            },
            {
                "origin": "rss_news",
                "current": {"thr": 0.50, "precision": 0.80, "recall": 0.40, "f1": 0.53, "labels": 15, "triggers": 6},
                "recommended": {"thr": 0.62, "precision": 0.82, "recall": 0.42, "f1": 0.56, "labels": 15, "triggers": 5},
                "risk": ["ok"],
            },
        ],
    }
    _write(models / "threshold_backtest.json", bt)

    # Run section
    from scripts.summary_sections import threshold_auto_apply as taa
    importlib.reload(taa)

    ctx = SummaryContext(
        logs_dir=tmp_path / "logs",
        models_dir=models,
        is_demo=False,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )
    md: list[str] = []
    taa.append(md, ctx)

    # Markdown sanity
    text = "\n".join(md)
    assert "Threshold Auto-Apply" in text
    assert "applied" in text
    assert "skipped" in text

    # Threshold file updated for applied origin only
    new_thr = json.loads((models / "per_origin_thresholds.json").read_text(encoding="utf-8"))
    # reddit applied to 0.56
    assert abs(new_thr.get("reddit", 0.0) - 0.56) < 1e-6
    # twitter stayed 0.50 due to risk flag
    assert abs(new_thr.get("twitter", 0.0) - 0.50) < 1e-6
    # rss_news stayed 0.50 due to large jump
    assert abs(new_thr.get("rss_news", 0.50) - 0.50) < 1e-6

    # Artifact written
    aa = json.loads((models / "threshold_auto_apply.json").read_text(encoding="utf-8"))
    assert isinstance(aa.get("per_origin"), list) and len(aa["per_origin"]) >= 3
    assert aa["guardrails"]["min_precision"] == 0.75
    assert aa["guardrails"]["max_delta"] == 0.10


def test_demo_seed_when_empty(monkeypatch, tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DEMO_MODE", "true")

    # No inputs → demo path
    from scripts.summary_sections import threshold_auto_apply as taa
    importlib.reload(taa)

    ctx = SummaryContext(
        logs_dir=tmp_path / "logs",
        models_dir=models,
        is_demo=True,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )
    md: list[str] = []
    taa.append(md, ctx)

    # Artifact exists and marked demo
    j = json.loads((models / "threshold_auto_apply.json").read_text(encoding="utf-8"))
    assert j.get("demo", False) is True
    assert isinstance(j.get("per_origin"), list) and len(j["per_origin"]) >= 2
