# tests/test_signal_quality_trend_chart.py
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, importlib

from scripts.summary_sections.common import SummaryContext
import scripts.summary_sections.signal_quality as sq

def _iso(dt): return dt.isoformat().replace("+00:00","Z")

def test_signal_quality_trend_image_is_written(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_SIGNAL_WINDOW_H", "72")
    monkeypatch.setenv("MW_SIGNAL_BATCH_H", "3")

    # Ensure artifacts dir exists (writer will also mkdir)
    (tmp_path / "artifacts").mkdir(exist_ok=True)

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    meta = {
        "window_hours": 72,
        "batch_hours": 3,
        "generated_at": _iso(now),
        "batches": [
            {
                "start": _iso(now - timedelta(hours=9)),
                "end":   _iso(now - timedelta(hours=6)),
                "triggers": 6, "true": 3, "false": 3,
                "precision": 0.5, "class": "Mixed", "emoji": "⚠️", "demo": False,
            },
            {
                "start": _iso(now - timedelta(hours=6)),
                "end":   _iso(now - timedelta(hours=3)),
                "triggers": 5, "true": 4, "false": 1,
                "precision": 0.8, "class": "Strong", "emoji": "✅", "demo": False,
            },
            {
                "start": _iso(now - timedelta(hours=3)),
                "end":   _iso(now),
                "triggers": 4, "true": 1, "false": 3,
                "precision": 0.25, "class": "Weak", "emoji": "❌", "demo": False,
            },
        ],
        "demo": False,
    }
    (models / "signal_quality_summary.json").write_text(json.dumps(meta), encoding="utf-8")

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
    importlib.reload(sq)
    sq.append(md, ctx, write_json=False)

    text = "\n".join(md)
    # Header present
    assert "Signal Quality Summary (last 72h, 3h buckets)" in text
    # Chart referenced
    assert "signal_quality_trend_72h.png" in text

    # Image exists in repo artifacts/
    img_path = Path("artifacts") / "signal_quality_trend_72h.png"
    assert img_path.exists() and img_path.stat().st_size > 0
