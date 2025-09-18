# tests/test_suppression_rate_by_origin.py
from __future__ import annotations

import importlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.summary_sections.common import SummaryContext

def _iso(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def test_suppression_math_and_artifact(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_SUPPRESSION_WINDOW_H", "48")
    monkeypatch.setenv("MW_TRIGGER_JOIN_MIN", "5")

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    # --- Seed candidates (within window) ---
    cand = []
    # reddit: 16 candidates (every 2 min around now-1h)
    for i in range(16):
        cand.append({"timestamp": _iso(now - timedelta(hours=1, minutes=i*2)), "origin": "reddit"})
    # twitter: 10 candidates (now-4h)
    for i in range(10):
        cand.append({"timestamp": _iso(now - timedelta(hours=4, minutes=i*3)), "origin": "twitter"})
    # rss_news: 8 candidates (now-5h)
    for i in range(8):
        cand.append({"timestamp": _iso(now - timedelta(hours=5, minutes=i*4)), "origin": "rss_news"})

    (logs / "candidates.jsonl").write_text("\n".join(json.dumps(r) for r in cand), encoding="utf-8")

    # --- Seed triggers close to some candidates (<= 5 min) ---
    trig = []
    # reddit: 5 triggers → suppression 11/16 = 68.75% -> Medium
    for i in range(5):
        trig.append({"timestamp": _iso(now - timedelta(hours=1, minutes=i*2+1)), "origin": "reddit"})
    # twitter: 1 trigger → suppression 9/10 = 90% -> High
    trig.append({"timestamp": _iso(now - timedelta(hours=4, minutes=1)), "origin": "twitter"})
    # rss_news: 6 triggers → suppression 2/8 = 25% -> Low
    for i in range(6):
        trig.append({"timestamp": _iso(now - timedelta(hours=5, minutes=i*4+1)), "origin": "rss_news"})

    (models / "trigger_history.jsonl").write_text("\n".join(json.dumps(r) for r in trig), encoding="utf-8")

    from scripts.summary_sections import suppression_rate_by_origin as sro
    importlib.reload(sro)

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
    sro.append(md, ctx)

    out_md = "\n".join(md)
    assert "Suppression Rate by Origin (48h)" in out_md
    assert "reddit" in out_md and "Medium" in out_md
    assert "twitter" in out_md and "High" in out_md
    assert "rss_news" in out_md and "Low" in out_md

    # Artifact exists & schema basics
    art = json.loads((models / "suppression_rate_per_origin.json").read_text(encoding="utf-8"))
    assert art["window_hours"] == 48 and art["join_minutes"] == 5
    po = {r["origin"]: r for r in art["per_origin"]}
    assert po["reddit"]["candidates"] == 16 and po["reddit"]["triggers"] == 5 and po["reddit"]["suppressed"] == 11
    assert po["twitter"]["candidates"] == 10 and po["twitter"]["suppressed"] == 9
    assert po["rss_news"]["candidates"] == 8 and po["rss_news"]["suppressed"] == 2


def test_demo_seed_when_sparse(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_SUPPRESSION_WINDOW_H", "48")
    monkeypatch.setenv("MW_TRIGGER_JOIN_MIN", "5")

    from scripts.summary_sections import suppression_rate_by_origin as sro
    importlib.reload(sro)

    ctx = SummaryContext(
        logs_dir=logs,
        models_dir=models,
        is_demo=True,  # trigger demo mode
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )

    md: list[str] = []
    sro.append(md, ctx)
    txt = "\n".join(md)
    assert "(demo)" in txt

    art = json.loads((models / "suppression_rate_per_origin.json").read_text(encoding="utf-8"))
    assert art["demo"] is True
    assert len(art["per_origin"]) >= 3