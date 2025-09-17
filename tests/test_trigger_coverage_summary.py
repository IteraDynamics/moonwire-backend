# tests/test_trigger_coverage_summary.py
import json
import importlib
from pathlib import Path
from datetime import datetime, timedelta, timezone

from scripts.summary_sections.common import SummaryContext, _iso
from scripts.summary_sections import trigger_coverage_summary as tcs


def _write_jsonl(p: Path, rows):
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_trigger_coverage_counts_and_classes(tmp_path, monkeypatch):
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_TRIGGER_COVERAGE_WINDOW_H", "48")
    monkeypatch.setenv("MW_TRIGGER_JOIN_MIN", "5")

    now = datetime.now(timezone.utc)

    # --- candidates in logs/ (require burst_z to be picked up) ---
    cand_rows = []
    # twitter 11 candidates (we will match 2 triggers -> ~18.2% High)
    for m in [5, 10, 15, 20, 25, 30, 32, 34, 36, 38, 40]:
        cand_rows.append({"timestamp": _iso(now - timedelta(minutes=m)), "origin":"twitter", "burst_z": 2.1})
    # reddit 14 candidates (match 1 trigger -> ~7.1% Medium)
    for m in [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]:
        cand_rows.append({"timestamp": _iso(now - timedelta(minutes=m)), "origin":"reddit", "burst_z": 1.1})
    # rss_news 10 candidates (0 triggers -> 0% Low)
    for m in [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]:
        cand_rows.append({"timestamp": _iso(now - timedelta(minutes=m)), "origin":"rss_news", "burst_z": 0.9})

    _write_jsonl(logs / "candidates.jsonl", cand_rows)

    # --- triggers in models/trigger_history.jsonl ---
    trig_rows = [
        # twitter: two triggers near 10 and 34 minutes
        {"timestamp": _iso(now - timedelta(minutes=11)), "origin":"twitter", "decision":"triggered"},
        {"timestamp": _iso(now - timedelta(minutes=33)), "origin":"twitter", "decision":"triggered"},

        # reddit: one trigger near 60 minutes
        {"timestamp": _iso(now - timedelta(minutes=61)), "origin":"reddit", "decision":"triggered"},

        # rss_news: none
    ]
    _write_jsonl(models / "trigger_history.jsonl", trig_rows)

    # context + call
    importlib.reload(tcs)
    ctx = SummaryContext(
        logs_dir=logs,
        models_dir=models,
        is_demo=False,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )
    md = []
    tcs.append(md, ctx)

    out_text = "\n".join(md)
    assert "Trigger Coverage Summary (48h)" in out_text
    assert "`twitter`" in out_text and "High" in out_text
    assert "`reddit`" in out_text and "Medium" in out_text
    assert "`rss_news`" in out_text and "Low" in out_text

    # artifact
    artifact = models / "trigger_coverage_per_origin.json"
    assert artifact.exists()
    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert data["window_hours"] == 48
    assert isinstance(data["per_origin"], list) and len(data["per_origin"]) >= 3
    one = {r["origin"]: r for r in data["per_origin"]}
    assert one["twitter"]["triggers"] == 2
    assert one["reddit"]["triggers"] == 1
    assert one["rss_news"]["triggers"] == 0


def test_trigger_coverage_demo_fallback(tmp_path, monkeypatch):
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_TRIGGER_COVERAGE_WINDOW_H", "48")
    monkeypatch.setenv("MW_TRIGGER_JOIN_MIN", "5")

    importlib.reload(tcs)
    ctx = SummaryContext(
        logs_dir=logs,
        models_dir=models,
        is_demo=True,   # no data → should seed demo
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )
    md = []
    tcs.append(md, ctx)

    txt = "\n".join(md)
    assert "(demo)" in txt

    payload = json.loads((models / "trigger_coverage_per_origin.json").read_text(encoding="utf-8"))
    assert payload["demo"] is True
    assert len(payload["per_origin"]) >= 3
