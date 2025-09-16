# tests/test_threshold_quality_per_origin.py
from datetime import datetime, timezone, timedelta
from pathlib import Path
import importlib
import json
import os

from scripts.summary_sections.common import SummaryContext
import scripts.summary_sections.threshold_quality_per_origin as tqpo


def _iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def test_threshold_quality_counts_and_classes(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_SCORE_WINDOW_H", "48")
    monkeypatch.setenv("MW_THRESHOLD_JOIN_MIN", "5")
    monkeypatch.setenv("MW_THRESHOLD_MIN_LABELS", "3")

    now = datetime.now(timezone.utc)

    # --- Triggers with scores per origin ---
    trig = [
        # reddit (thr=0.50) => Strong: TP=3, FP=0, FN=1 -> F1=0.86
        {"timestamp": _iso(now - timedelta(minutes=50)), "origin": "reddit", "adjusted_score": 0.80},
        {"timestamp": _iso(now - timedelta(minutes=48)), "origin": "reddit", "adjusted_score": 0.72},
        {"timestamp": _iso(now - timedelta(minutes=46)), "origin": "reddit", "adjusted_score": 0.62},
        {"timestamp": _iso(now - timedelta(minutes=44)), "origin": "reddit", "adjusted_score": 0.30},

        # twitter (thr=0.42) => Mixed: TP=1, FP=1, FN=1 -> F1=0.50
        {"timestamp": _iso(now - timedelta(minutes=40)), "origin": "twitter", "adjusted_score": 0.55},
        {"timestamp": _iso(now - timedelta(minutes=38)), "origin": "twitter", "adjusted_score": 0.50},
        {"timestamp": _iso(now - timedelta(minutes=36)), "origin": "twitter", "adjusted_score": 0.20},

        # rss_news (thr=0.50) => Weak: TP=0, FP=2, FN=1 -> F1=0.00
        {"timestamp": _iso(now - timedelta(minutes=32)), "origin": "rss_news", "adjusted_score": 0.60},
        {"timestamp": _iso(now - timedelta(minutes=30)), "origin": "rss_news", "adjusted_score": 0.70},
        {"timestamp": _iso(now - timedelta(minutes=28)), "origin": "rss_news", "adjusted_score": 0.40},
    ]
    with (models / "trigger_history.jsonl").open("w", encoding="utf-8") as f:
        for r in trig:
            f.write(json.dumps(r) + "\n")

    # --- Labels close in time ---
    labs = [
        # reddit: T, T, T near high scores; T near low score (FN)
        {"timestamp": _iso(now - timedelta(minutes=50)), "origin": "reddit", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=48)), "origin": "reddit", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=46)), "origin": "reddit", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=44)), "origin": "reddit", "label": True},

        # twitter: TP (true near 0.55), FP (false near 0.50), FN (true near 0.20)
        {"timestamp": _iso(now - timedelta(minutes=40)), "origin": "twitter", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=38)), "origin": "twitter", "label": False},
        {"timestamp": _iso(now - timedelta(minutes=36)), "origin": "twitter", "label": True},

        # rss_news: FP, FP (false near >= thr) and FN (true near < thr)
        {"timestamp": _iso(now - timedelta(minutes=32)), "origin": "rss_news", "label": False},
        {"timestamp": _iso(now - timedelta(minutes=30)), "origin": "rss_news", "label": False},
        {"timestamp": _iso(now - timedelta(minutes=28)), "origin": "rss_news", "label": True},
    ]
    with (models / "label_feedback.jsonl").open("w", encoding="utf-8") as f:
        for r in labs:
            f.write(json.dumps(r) + "\n")

    # Per-origin thresholds
    (models / "per_origin_thresholds.json").write_text(
        json.dumps({"reddit": 0.50, "twitter": 0.42, "rss_news": 0.50}), encoding="utf-8"
    )

    # Context
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
    importlib.reload(tqpo)
    tqpo.append(md, ctx)

    out_text = "\n".join(md)
    assert "📊 Per-Origin Threshold Quality (48h)" in out_text
    assert "`reddit`" in out_text and "Strong" in out_text
    assert "`twitter`" in out_text and "Mixed" in out_text
    assert "`rss_news`" in out_text and "Weak" in out_text

    # Artifact file exists and has expected counts
    out_path = models / "threshold_quality_per_origin.json"
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    per_origin = {d["origin"]: d for d in data["per_origin"]}
    assert per_origin["reddit"]["tp"] == 3 and per_origin["reddit"]["fn"] == 1 and per_origin["reddit"]["fp"] == 0
    assert per_origin["twitter"]["tp"] == 1 and per_origin["twitter"]["fn"] == 1 and per_origin["twitter"]["fp"] == 1
    assert per_origin["rss_news"]["tp"] == 0 and per_origin["rss_news"]["fn"] == 1 and per_origin["rss_news"]["fp"] == 2


def test_demo_fallback_when_sparse(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_THRESHOLD_MIN_LABELS", "3")

    # Leave logs empty to force demo seeding
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
    importlib.reload(tqpo)
    tqpo.append(md, ctx)

    out_text = "\n".join(md)
    assert "(demo)" in out_text
    data = json.loads((models / "threshold_quality_per_origin.json").read_text(encoding="utf-8"))
    assert data["demo"] is True
    assert len(data["per_origin"]) >= 1