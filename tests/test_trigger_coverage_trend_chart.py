# tests/test_trigger_coverage_trend_chart.py
import json, importlib
from pathlib import Path
from datetime import datetime, timezone, timedelta

from scripts.summary_sections.common import SummaryContext

def _iso(dt):  # local helper
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def test_trigger_coverage_trend_chart(monkeypatch, tmp_path: Path):
    # Env + dirs
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_TRIGGER_COVERAGE_WINDOW_H", "48")
    monkeypatch.setenv("MW_TRIGGER_BUCKET_H", "3")

    # Seed candidates across buckets for two origins
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    cand_rows = [
        # bucket 1 (now-1h): twitter 3, reddit 1
        {"timestamp": _iso(now - timedelta(hours=1)), "origin": "twitter"},
        {"timestamp": _iso(now - timedelta(hours=1, minutes=10)), "origin": "twitter"},
        {"timestamp": _iso(now - timedelta(hours=1, minutes=20)), "origin": "twitter"},
        {"timestamp": _iso(now - timedelta(hours=1, minutes=30)), "origin": "reddit"},

        # bucket 2 (now-4h): twitter 2, reddit 4
        {"timestamp": _iso(now - timedelta(hours=4)), "origin": "twitter"},
        {"timestamp": _iso(now - timedelta(hours=4, minutes=5)), "origin": "twitter"},
        {"timestamp": _iso(now - timedelta(hours=4, minutes=10)), "origin": "reddit"},
        {"timestamp": _iso(now - timedelta(hours=4, minutes=15)), "origin": "reddit"},
        {"timestamp": _iso(now - timedelta(hours=4, minutes=20)), "origin": "reddit"},
        {"timestamp": _iso(now - timedelta(hours=4, minutes=25)), "origin": "reddit"},
    ]
    (logs / "candidates.jsonl").write_text("\n".join(json.dumps(r) for r in cand_rows), encoding="utf-8")

    # Seed triggers within those buckets
    trig_rows = [
        {"timestamp": _iso(now - timedelta(hours=1, minutes=5)), "origin": "twitter"},
        {"timestamp": _iso(now - timedelta(hours=4, minutes=3)), "origin": "reddit"},
        {"timestamp": _iso(now - timedelta(hours=4, minutes=8)), "origin": "reddit"},
    ]
    (models / "trigger_history.jsonl").write_text("\n".join(json.dumps(r) for r in trig_rows), encoding="utf-8")

    # Run the section
    from scripts.summary_sections import trigger_coverage_trend as tct
    importlib.reload(tct)

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
    tct.append(md, ctx)

    # Assertions
    out = "\n".join(md)
    assert "Trigger Coverage Trend (48h)" in out

    img = tmp_path / "artifacts" / "trigger_coverage_trend_48h.png"
    assert img.exists() and img.stat().st_size > 0