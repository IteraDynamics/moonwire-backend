# tests/test_threshold_recommendations.py
from pathlib import Path
from datetime import datetime, timezone, timedelta
import importlib, json, os

from scripts.summary_sections.common import SummaryContext, _iso

def test_threshold_reco_basic_and_guardrail(monkeypatch, tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_THR_RECO_WINDOW_H", "72")
    monkeypatch.setenv("MW_THRESHOLD_JOIN_MIN", "5")
    monkeypatch.setenv("MW_THR_RECO_MIN_LABELS", "4")
    monkeypatch.setenv("MW_THR_RECO_MIN_PREC", "0.75")
    monkeypatch.setenv("MW_THR_RECO_STRATEGY", "precision_min_recall_max")
    monkeypatch.setenv("MW_THR_RECO_MAX_DELTA", "0.10")
    monkeypatch.setenv("MW_THR_RECO_ALLOW_LARGE_JUMP", "false")

    # Current thresholds
    (models / "per_origin_thresholds.json").write_text(json.dumps({"reddit": 0.50, "twitter": 0.50}), encoding="utf-8")

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    # Triggers with scores close to labels
    trig = [
        # reddit scores (should push recommended to ~0.56 but clamp to +0.10 if needed)
        {"timestamp": _iso(now - timedelta(minutes=50)), "origin": "reddit",  "adjusted_score": 0.60},
        {"timestamp": _iso(now - timedelta(minutes=49)), "origin": "reddit",  "adjusted_score": 0.58},
        {"timestamp": _iso(now - timedelta(minutes=48)), "origin": "reddit",  "adjusted_score": 0.40},
        {"timestamp": _iso(now - timedelta(minutes=47)), "origin": "reddit",  "adjusted_score": 0.52},
        # twitter scores
        {"timestamp": _iso(now - timedelta(minutes=40)), "origin": "twitter", "adjusted_score": 0.47},
        {"timestamp": _iso(now - timedelta(minutes=39)), "origin": "twitter", "adjusted_score": 0.46},
        {"timestamp": _iso(now - timedelta(minutes=38)), "origin": "twitter", "adjusted_score": 0.30},
        {"timestamp": _iso(now - timedelta(minutes=37)), "origin": "twitter", "adjusted_score": 0.80},
    ]
    (models / "trigger_history.jsonl").write_text("\n".join(json.dumps(r) for r in trig), encoding="utf-8")

    # Labels aligned near triggers
    labs = [
        # reddit labels → True near higher scores, False near 0.40
        {"timestamp": _iso(now - timedelta(minutes=50)), "origin": "reddit",  "label": True},
        {"timestamp": _iso(now - timedelta(minutes=49)), "origin": "reddit",  "label": True},
        {"timestamp": _iso(now - timedelta(minutes=48)), "origin": "reddit",  "label": False},
        {"timestamp": _iso(now - timedelta(minutes=47)), "origin": "reddit",  "label": True},
        # twitter labels → mix; desire precision≥0.75 feasible near ~0.46-0.47 threshold
        {"timestamp": _iso(now - timedelta(minutes=40)), "origin": "twitter", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=39)), "origin": "twitter", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=38)), "origin": "twitter", "label": False},
        {"timestamp": _iso(now - timedelta(minutes=37)), "origin": "twitter", "label": True},
    ]
    (models / "label_feedback.jsonl").write_text("\n".join(json.dumps(r) for r in labs), encoding="utf-8")

    from scripts.summary_sections import threshold_recommendations as thr
    importlib.reload(thr)

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
    thr.append(md, ctx)

    # Markdown assertions
    out = "\n".join(md)
    assert "Threshold Recommendations" in out
    assert "objective" in out
    assert "`reddit`" in out
    assert "`twitter`" in out

    # JSON artifact exists and has schema-ish fields
    art = json.loads((models / "threshold_recommendations.json").read_text(encoding="utf-8"))
    assert art.get("window_hours") == 72
    assert isinstance(art.get("per_origin"), list) and len(art["per_origin"]) >= 2

    ro = {r["origin"]: r for r in art["per_origin"]}
    assert "reddit" in ro and "twitter" in ro
    # guardrail respected: delta magnitude <= 0.10 when allow_large_jump=false
    assert abs(ro["reddit"].get("delta") or 0) <= 0.10
    # recommended present and metrics computed
    assert ro["twitter"]["recommended"] is not None
    assert "precision" in ro["twitter"] and "recall" in ro["twitter"] and "f1" in ro["twitter"]

def test_threshold_reco_insufficient_and_demo(monkeypatch, tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_THR_RECO_WINDOW_H", "72")
    monkeypatch.setenv("MW_THR_RECO_MIN_LABELS", "10")
    monkeypatch.setenv("DEMO_MODE", "true")

    # Empty logs → demo seed expected
    from scripts.summary_sections import threshold_recommendations as thr
    importlib.reload(thr)

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
    thr.append(md, ctx)

    out = "\n".join(md)
    assert "Threshold Recommendations" in out
    assert "(demo)" in out

    art = json.loads((models / "threshold_recommendations.json").read_text(encoding="utf-8"))
    assert art.get("demo") is True
    assert len(art.get("per_origin", [])) >= 2
