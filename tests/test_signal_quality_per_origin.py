# tests/test_signal_quality_per_origin.py
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, importlib, os

from scripts.summary_sections.common import SummaryContext
from scripts.summary_sections import signal_quality_per_origin as sqpo

def _iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def test_per_origin_grouping_and_classification(tmp_path: Path, monkeypatch):
    # Point MODELS_DIR to temp
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("MW_SIGNAL_WINDOW_H", "72")
    monkeypatch.setenv("MW_SIGNAL_JOIN_MIN", "5")

    now = datetime.now(timezone.utc)

    # Triggers (only 'triggered' count) across three origins
    trig = [
        {"timestamp": _iso(now - timedelta(minutes=50)), "origin": "reddit",   "decision": "triggered"},
        {"timestamp": _iso(now - timedelta(minutes=49)), "origin": "reddit",   "decision": "triggered"},
        {"timestamp": _iso(now - timedelta(minutes=48)), "origin": "reddit",   "decision": "triggered"},
        {"timestamp": _iso(now - timedelta(minutes=47)), "origin": "reddit",   "decision": "triggered"},
        {"timestamp": _iso(now - timedelta(minutes=46)), "origin": "twitter",  "decision": "triggered"},
        {"timestamp": _iso(now - timedelta(minutes=45)), "origin": "twitter",  "decision": "triggered"},
        {"timestamp": _iso(now - timedelta(minutes=44)), "origin": "twitter",  "decision": "triggered"},
        {"timestamp": _iso(now - timedelta(minutes=43)), "origin": "rss_news", "decision": "triggered"},
        {"timestamp": _iso(now - timedelta(minutes=42)), "origin": "rss_news", "decision": "triggered"},
    ]
    with (models_dir / "trigger_history.jsonl").open("w", encoding="utf-8") as f:
        for r in trig:
            f.write(json.dumps(r) + "\n")

    # Labels close in time; reddit: 3T/1F (precision=0.75 => Strong)
    # twitter: 1T/2F (precision=0.33 => Weak)
    # rss_news: 1T/1F (precision=0.5, n=2 => Mixed)
    labs = [
        {"timestamp": _iso(now - timedelta(minutes=49)), "origin": "reddit",   "label": True},
        {"timestamp": _iso(now - timedelta(minutes=48)), "origin": "reddit",   "label": True},
        {"timestamp": _iso(now - timedelta(minutes=47)), "origin": "reddit",   "label": True},
        {"timestamp": _iso(now - timedelta(minutes=46)), "origin": "reddit",   "label": False},

        {"timestamp": _iso(now - timedelta(minutes=45)), "origin": "twitter",  "label": True},
        {"timestamp": _iso(now - timedelta(minutes=44)), "origin": "twitter",  "label": False},
        {"timestamp": _iso(now - timedelta(minutes=43)), "origin": "twitter",  "label": False},

        {"timestamp": _iso(now - timedelta(minutes=42)), "origin": "rss_news", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=41)), "origin": "rss_news", "label": False},
    ]
    with (models_dir / "label_feedback.jsonl").open("w", encoding="utf-8") as f:
        for r in labs:
            f.write(json.dumps(r) + "\n")

    ctx = SummaryContext(
        logs_dir=tmp_path / "logs",
        models_dir=models_dir,
        is_demo=False,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)

    md = []
    importlib.reload(sqpo)
    sqpo.append(md, ctx)

    text = "\n".join(md)
    # Classification checks
    assert "reddit" in text and "Strong" in text
    assert "twitter" in text and "Weak" in text
    assert "rss_news" in text and "Mixed" in text

    # Artifact written
    art = models_dir / "signal_quality_per_origin.json"
    assert art.exists()
    blob = json.loads(art.read_text())
    origins = {r["origin"]: r for r in blob["per_origin"]}
    assert origins["reddit"]["precision"] == 0.75
    assert origins["twitter"]["precision"] == 1/3
    assert origins["rss_news"]["precision"] == 0.5

def test_demo_fallback_when_empty(tmp_path: Path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("DEMO_MODE", "true")

    ctx = SummaryContext(
        logs_dir=tmp_path / "logs",
        models_dir=models_dir,
        is_demo=True,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)

    md = []
    sqpo.append(md, ctx)
    text = "\n".join(md)
    assert "seeded demo data" in text

    art = models_dir / "signal_quality_per_origin.json"
    assert art.exists()
    blob = json.loads(art.read_text())
    assert blob.get("demo") is True
    assert len(blob.get("per_origin", [])) >= 3
