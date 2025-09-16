# tests/test_signal_quality_per_version_chart.py
import json, importlib
from pathlib import Path
from datetime import datetime, timedelta, timezone

from scripts.summary_sections.common import SummaryContext

def _iso(dt): return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def test_version_trend_chart_is_created(monkeypatch, tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_SIGNAL_WINDOW_H", "72")
    monkeypatch.setenv("MW_SIGNAL_VERSION_CHART", "true")

    # Seed artifact with a small time series for two versions
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    data = {
        "window_hours": 72,
        "join_minutes": 5,
        "generated_at": _iso(now),
        "per_version": [
            {"version": "v0.5.9", "precision": 0.70, "labels": 5, "class": "Mixed"},
            {"version": "v0.5.8", "precision": 0.80, "labels": 6, "class": "Strong"},
        ],
        "series": [
            {"version": "v0.5.9", "t": _iso(now - timedelta(hours=6)), "precision": 0.60},
            {"version": "v0.5.9", "t": _iso(now - timedelta(hours=3)), "precision": 0.68},
            {"version": "v0.5.9", "t": _iso(now),                     "precision": 0.70},
            {"version": "v0.5.8", "t": _iso(now - timedelta(hours=6)), "precision": 0.76},
            {"version": "v0.5.8", "t": _iso(now - timedelta(hours=3)), "precision": 0.79},
            {"version": "v0.5.8", "t": _iso(now),                     "precision": 0.80},
        ],
        "demo": False,
    }
    (models / "signal_quality_per_version.json").write_text(json.dumps(data), encoding="utf-8")

    # Ensure artifacts dir exists for the run
    (tmp_path / "artifacts").mkdir(exist_ok=True)

    # Call the section
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

    txt = "\n".join(md)
    assert "Per-Version Signal Quality" in txt
    # trend line presence
    assert "Per-Version Precision Trend" in txt

    # image was written
    img = tmp_path / "artifacts" / "signal_quality_by_version_72h.png"
    assert img.exists() and img.stat().st_size > 0