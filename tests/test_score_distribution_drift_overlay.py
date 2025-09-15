# tests/test_score_distribution_drift_overlay.py
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, importlib
import os

from scripts.summary_sections.common import SummaryContext
from scripts.summary_sections import score_distribution


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def test_score_distribution_overlay_creates_image_and_counts(tmp_path: Path, monkeypatch):
    # Wire models dir so the section reads our temp trigger_history.jsonl
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models_dir))

    # Build recent rows with drifted vs non-drifted
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp": _iso(now - timedelta(hours=1)), "adjusted_score": 0.10, "drifted_features": ["burst_z"]},
        {"timestamp": _iso(now - timedelta(hours=2)), "adjusted_score": 0.20, "drifted_features": []},
        {"timestamp": _iso(now - timedelta(hours=3)), "adjusted_score": 0.40, "drift": True},
        {"timestamp": _iso(now - timedelta(hours=4)), "adjusted_score": 0.60, "drifted": False},
        {"timestamp": _iso(now - timedelta(hours=5)), "adjusted_score": 0.80, "drifted_features": []},
    ]
    hist_path = models_dir / "trigger_history.jsonl"
    with hist_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Build a minimal context
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
    # import fresh module (not strictly necessary, but consistent)
    importlib.reload(score_distribution)
    score_distribution.append(md, ctx, hours=48, min_points=1)

    joined = "\n".join(md)
    # We expect the split counts line to appear
    assert "split: drifted=" in joined

    # And the image file to exist
    overlay = Path("artifacts") / "score_hist_drift_overlay_48h.png"
    assert overlay.exists()