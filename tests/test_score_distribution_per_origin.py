# tests/test_score_distribution_per_origin.py
import importlib, json, os
from pathlib import Path
from datetime import datetime, timezone, timedelta

from scripts.summary_sections.common import SummaryContext
import scripts.summary_sections.score_distribution_per_origin as sdpo

def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def test_per_origin_overlay_writes_images_and_stats(tmp_path, monkeypatch):
    # MODELS_DIR → temp
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("MW_SCORE_WINDOW_H", "48")

    # Ensure artifacts dir is writable in repo root of test run
    (tmp_path / "artifacts").mkdir(exist_ok=True)

    # Build recent trigger history across two origins with drift mix
    now = datetime.now(timezone.utc)

    rows = [
        # reddit (3 points: 2 drifted, 1 non)
        {"timestamp": _iso(now - timedelta(hours=1)), "origin": "reddit", "adjusted_score": 0.80, "drifted_features": ["burst_z"]},
        {"timestamp": _iso(now - timedelta(hours=1, minutes=10)), "origin": "reddit", "adjusted_score": 0.60, "drift": True},
        {"timestamp": _iso(now - timedelta(hours=1, minutes=20)), "origin": "reddit", "adjusted_score": 0.20},

        # twitter (3 points: 1 drifted, 2 non)
        {"timestamp": _iso(now - timedelta(hours=2)), "origin": "twitter", "adjusted_score": 0.15},
        {"timestamp": _iso(now - timedelta(hours=2, minutes=5)), "origin": "twitter", "adjusted_score": 0.35},
        {"timestamp": _iso(now - timedelta(hours=2, minutes=10)), "origin": "twitter", "adjusted_score": 0.55, "drifted": True},
    ]

    with (models_dir / "trigger_history.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Context
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
    importlib.reload(sdpo)
    sdpo.append(md, ctx)

    text = "\n".join(md)

    # Basic header appears
    assert "📐 Score Distribution by Origin (48h)" in text

    # Stats lines for each origin
    assert "reddit" in text and "split: drifted=" in text
    assert "twitter" in text and "split: drifted=" in text

    # Image artifacts per origin exist
    from scripts.summary_sections.score_distribution_per_origin import _slug
    reddit_img = Path("artifacts") / f"score_hist_{_slug('reddit')}_overlay.png"
    twitter_img = Path("artifacts") / f"score_hist_{_slug('twitter')}_overlay.png"
    assert reddit_img.exists(), f"Missing {reddit_img}"
    assert twitter_img.exists(), f"Missing {twitter_img}"