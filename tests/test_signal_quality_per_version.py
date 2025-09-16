# tests/test_signal_quality_per_version.py
from datetime import datetime, timezone, timedelta
from pathlib import Path
import importlib, json, os

from scripts.summary_sections.common import SummaryContext

def _iso(dt): return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

def test_per_version_grouping_and_artifact(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_SIGNAL_WINDOW_H", "72")
    monkeypatch.setenv("MW_SIGNAL_JOIN_MIN", "5")
    monkeypatch.setenv("MW_THRESHOLD_MIN_LABELS", "2")

    now = datetime.now(timezone.utc)

    # Triggers across two versions (timestamps near labels)
    trig = [
        {"timestamp": _iso(now - timedelta(minutes=50)), "origin": "twitter",  "model_version": "v0.5.9"},
        {"timestamp": _iso(now - timedelta(minutes=48)), "origin": "twitter",  "model_version": "v0.5.9"},
        {"timestamp": _iso(now - timedelta(minutes=40)), "origin": "reddit",   "model_version": "v0.5.8"},
        {"timestamp": _iso(now - timedelta(minutes=39)), "origin": "reddit",   "model_version": "v0.5.8"},
    ]
    (models / "trigger_history.jsonl").write_text("\n".join(json.dumps(r) for r in trig), encoding="utf-8")

    # Labels: twitter → T,F (precision 0.5), reddit → T + unmatched true (FN) to push recall down
    labs = [
        {"timestamp": _iso(now - timedelta(minutes=50)), "origin": "twitter", "label": True,  "model_version": "v0.5.9"},
        {"timestamp": _iso(now - timedelta(minutes=48)), "origin": "twitter", "label": False, "model_version": "v0.5.9"},

        {"timestamp": _iso(now - timedelta(minutes=40)), "origin": "reddit",  "label": True,  "model_version": "v0.5.8"},
        {"timestamp": _iso(now - timedelta(minutes=30)), "origin": "reddit",  "label": True,  "model_version": "v0.5.8"},  # FN (no nearby trigger)
    ]
    (models / "label_feedback.jsonl").write_text("\n".join(json.dumps(r) for r in labs), encoding="utf-8")

    from scripts.summary_sections import signal_quality_per_version as sqpv
    importlib.reload(sqpv)

    ctx = SummaryContext(
        logs_dir=tmp_path / "logs",
        models_dir=models,
        is_demo=False,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)

    md = []
    sqpv.append(md, ctx)

    out = "\n".join(md)
    assert "Per-Version Signal Quality" in out
    assert "`v0.5.9`" in out
    assert "`v0.5.8`" in out

    # artifact exists and schema-ish
    art = models / "signal_quality_per_version.json"
    assert art.exists()
    js = json.loads(art.read_text())
    assert "per_version" in js and isinstance(js["per_version"], list)
    versions = {r["version"] for r in js["per_version"]}
    assert {"v0.5.9", "v0.5.8"} <= versions