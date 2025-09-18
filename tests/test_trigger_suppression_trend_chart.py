# tests/test_trigger_suppression_trend_chart.py
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json, importlib

from scripts.summary_sections.common import SummaryContext, _iso

def test_trigger_suppression_trend_chart(monkeypatch, tmp_path: Path):
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_TRIGGER_SUPPRESSION_WINDOW_H", "48")
    monkeypatch.setenv("MW_TRIGGER_BUCKET_H", "3")

    # Seed candidates across two buckets for two origins
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    cand_rows = [
        # bucket ~ now-1h
        {"timestamp": _iso(now - timedelta(hours=1)), "origin": "twitter"},
        {"timestamp": _iso(now - timedelta(hours=1, minutes=10)), "origin": "twitter"},
        {"timestamp": _iso(now - timedelta(hours=1, minutes=20)), "origin": "reddit"},
        # bucket ~ now-4h
        {"timestamp": _iso(now - timedelta(hours=4, minutes= 0)), "origin": "twitter"},
        {"timestamp": _iso(now - timedelta(hours=4, minutes=10)), "origin": "reddit"},
        {"timestamp": _iso(now - timedelta(hours=4, minutes=20)), "origin": "reddit"},
    ]
    (logs / "candidates.jsonl").write_text("\n".join(json.dumps(r) for r in cand_rows), encoding="utf-8")

    trig_rows = [
        {"timestamp": _iso(now - timedelta(hours=1, minutes=5)), "origin": "twitter"},  # matches bucket ~now-1h
        {"timestamp": _iso(now - timedelta(hours=4, minutes=5)), "origin": "reddit"},   # matches bucket ~now-4h
    ]
    (models / "trigger_history.jsonl").write_text("\n".join(json.dumps(r) for r in trig_rows), encoding="utf-8")

    from scripts.summary_sections import trigger_suppression_trend as tst
    importlib.reload(tst)

    ctx = SummaryContext(
        logs_dir=logs,
        models_dir=models,
        is_demo=False,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )

    md: list[str] = []
    tst.append(md, ctx)
    out = "\n".join(md)

    # Markdown references
    assert "Trigger Suppression Trend (48h)" in out

    img = tmp_path / "artifacts" / "trigger_suppression_trend_48h.png"
    assert img.exists() and img.stat().st_size > 0
