import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.analytics.source_metrics import compute_source_metrics
from main import app

client = TestClient(app)

def make_log(ts: datetime, origin: str) -> dict:
    return {
        "timestamp": ts.isoformat(),
        "origin": origin
    }

def test_source_metrics_happy_path(tmp_path: Path):
    now = datetime.now(timezone.utc)

    flags = [make_log(now, "twitter")] * 5 + [make_log(now, "reddit")] * 3
    triggers = [make_log(now, "twitter")] * 2 + [make_log(now, "reddit")] * 1

    flags_path = tmp_path / "flags.jsonl"
    triggers_path = tmp_path / "triggers.jsonl"

    flags_path.write_text("\n".join(json.dumps(f) for f in flags))
    triggers_path.write_text("\n".join(json.dumps(t) for t in triggers))

    result = compute_source_metrics(flags_path, triggers_path, days=7, min_count=1)

    assert result["window_days"] == 7
    assert result["total_triggers"] == 3
    assert len(result["origins"]) == 2

    reddit = next(o for o in result["origins"] if o["origin"] == "reddit")
    assert reddit["flags"] == 3
    assert reddit["triggers"] == 1
    assert reddit["precision"] == 0.33
    assert reddit["recall"] == 0.33


def test_source_metrics_zero_data(tmp_path: Path):
    flags_path = tmp_path / "flags.jsonl"
    triggers_path = tmp_path / "triggers.jsonl"
    flags_path.write_text("")
    triggers_path.write_text("")

    result = compute_source_metrics(flags_path, triggers_path, days=7, min_count=1)

    assert result["origins"] == []
    assert result["total_triggers"] == 0


def test_source_metrics_demo_mode(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")

    flags_path = tmp_path / "flags.jsonl"
    triggers_path = tmp_path / "triggers.jsonl"
    flags_path.write_text("")
    triggers_path.write_text("")

    result = compute_source_metrics(flags_path, triggers_path, days=7, min_count=1)

    assert result["origins"]
    assert all(o["precision"] >= 0.0 for o in result["origins"])
    assert all(o["recall"] >= 0.0 for o in result["origins"])
    assert result["total_triggers"] > 0


def test_source_metrics_invalid_params():
    response = client.get("/internal/source-metrics?days=-1")
    assert response.status_code == 422  # FastAPI param validation

    response = client.get("/internal/source-metrics?min_count=-5")
    assert response.status_code == 422


def test_source_metrics_api_demo_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_MODE", "true")
    os.makedirs("logs", exist_ok=True)
    Path("logs/retraining_log.jsonl").write_text("")
    Path("logs/retraining_triggered.jsonl").write_text("")

    response = client.get("/internal/source-metrics?days=7&min_count=1")
    assert response.status_code == 200
    data = response.json()
    assert data["window_days"] == 7
    assert data["origins"]
    assert "origin" in data["origins"][0]
