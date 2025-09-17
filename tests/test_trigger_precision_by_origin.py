# tests/test_trigger_precision_by_origin.py
from datetime import datetime, timezone, timedelta
from pathlib import Path
import importlib, json, os

from scripts.summary_sections.common import SummaryContext

def _iso(dt): return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def test_trigger_precision_classes_and_artifact(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_TRIGGER_PRECISION_WINDOW_H", "48")
    monkeypatch.setenv("MW_TRIGGER_JOIN_MIN", "5")

    now = datetime.now(timezone.utc)

    # Triggers across three origins (timestamps near labels)
    trig = [
        # reddit (we'll make precision=1.0 with 3 labels => Strong)
        {"timestamp": _iso(now - timedelta(minutes=50)), "origin": "reddit"},
        {"timestamp": _iso(now - timedelta(minutes=48)), "origin": "reddit"},
        {"timestamp": _iso(now - timedelta(minutes=46)), "origin": "reddit"},

        # twitter (precision=0.5 with 4 labels => Mixed)
        {"timestamp": _iso(now - timedelta(minutes=40)), "origin": "twitter"},
        {"timestamp": _iso(now - timedelta(minutes=38)), "origin": "twitter"},
        {"timestamp": _iso(now - timedelta(minutes=36)), "origin": "twitter"},
        {"timestamp": _iso(now - timedelta(minutes=34)), "origin": "twitter"},

        # rss_news (precision<0.4 with 3+ labels => Weak)
        {"timestamp": _iso(now - timedelta(minutes=30)), "origin": "rss_news"},
        {"timestamp": _iso(now - timedelta(minutes=28)), "origin": "rss_news"},
        {"timestamp": _iso(now - timedelta(minutes=26)), "origin": "rss_news"},
    ]
    (models / "trigger_history.jsonl").write_text("\n".join(json.dumps(r) for r in trig), encoding="utf-8")

    # Labels to set the classes:
    labs = [
        # reddit: 3 true near those timestamps => precision 1.0 Strong
        {"timestamp": _iso(now - timedelta(minutes=50)), "origin": "reddit", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=48)), "origin": "reddit", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=46)), "origin": "reddit", "label": True},

        # twitter: 2 true, 2 false => 0.5 Mixed
        {"timestamp": _iso(now - timedelta(minutes=40)), "origin": "twitter", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=38)), "origin": "twitter", "label": False},
        {"timestamp": _iso(now - timedelta(minutes=36)), "origin": "twitter", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=34)), "origin": "twitter", "label": False},

        # rss_news: 0 true, 3 false => 0.0 Weak
        {"timestamp": _iso(now - timedelta(minutes=30)), "origin": "rss_news", "label": False},
        {"timestamp": _iso(now - timedelta(minutes=28)), "origin": "rss_news", "label": False},
        {"timestamp": _iso(now - timedelta(minutes=26)), "origin": "rss_news", "label": False},
    ]
    (models / "label_feedback.jsonl").write_text("\n".join(json.dumps(r) for r in labs), encoding="utf-8")

    from scripts.summary_sections import trigger_precision_by_origin as tpbo
    importlib.reload(tpbo)

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
    tpbo.append(md, ctx)
    out = "\n".join(md)

    assert "Trigger Precision by Origin" in out
    assert "`reddit`" in out and "Strong" in out
    assert "`twitter`" in out and "Mixed" in out
    assert "`rss_news`" in out and "Weak" in out

    # Artifact created & schema
    art = json.loads((models / "trigger_precision_by_origin.json").read_text(encoding="utf-8"))
    assert art["window_hours"] == 48
    assert art["join_minutes"] == 5
    assert isinstance(art["per_origin"], list) and len(art["per_origin"]) >= 3
    row = next(r for r in art["per_origin"] if r["origin"] == "reddit")
    assert row["precision"] == 1.0 and row["n"] == 3 and row["class"] == "Strong"

def test_trigger_precision_demo_seed(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_TRIGGER_PRECISION_WINDOW_H", "48")
    monkeypatch.setenv("MW_TRIGGER_JOIN_MIN", "5")

    from scripts.summary_sections import trigger_precision_by_origin as tpbo
    importlib.reload(tpbo)

    # No logs; demo mode on
    ctx = SummaryContext(
        logs_dir=tmp_path / "logs",
        models_dir=models,
        is_demo=True,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)

    md = []
    tpbo.append(md, ctx)
    txt = "\n".join(md)
    assert "(demo)" in txt
    data = json.loads((models / "trigger_precision_by_origin.json").read_text(encoding="utf-8"))
    assert data["demo"] is True and len(data["per_origin"]) >= 3