import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.analytics.origin_trends import compute_origin_trends
from main import app
import src.paths as paths


def _j(thing):
    return json.dumps(thing, separators=(",", ":"))


def _log(ts: datetime, origin: str):
    return {"timestamp": ts.isoformat(), "origin": origin}


def test_compute_origin_trends_day(tmp_path: Path):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    flags = [
        _log(now - timedelta(days=1), "twitter"),
        _log(now - timedelta(days=1), "twitter"),
        _log(now - timedelta(days=1), "reddit"),
        _log(now, "twitter"),
    ]
    triggers = [
        _log(now - timedelta(days=1), "twitter"),
        _log(now - timedelta(days=1), "reddit"),
    ]

    fpath = tmp_path / "flags.jsonl"
    tpath = tmp_path / "triggers.jsonl"
    fpath.write_text("\n".join(_j(r) for r in flags))
    tpath.write_text("\n".join(_j(r) for r in triggers))

    out = compute_origin_trends(fpath, tpath, days=7, interval="day")
    origins = {o["origin"]: o["buckets"] for o in out["origins"]}

    # twitter should have two day buckets
    tw = origins["twitter"]
    assert len(tw) == 2
    assert tw[0]["flags_count"] == 2  # yesterday
    assert tw[0]["triggers_count"] == 1
    assert tw[1]["flags_count"] == 1  # today
    assert tw[1]["triggers_count"] == 0

    # reddit should have one bucket yesterday
    rd = origins["reddit"]
    assert len(rd) == 1
    assert rd[0]["flags_count"] == 1
    assert rd[0]["triggers_count"] == 1


def test_compute_origin_trends_hour_vs_day(tmp_path: Path):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # two hours apart (same day)
    flags = [
        _log(now - timedelta(hours=2), "rss_news"),
        _log(now - timedelta(hours=1), "rss_news"),
    ]
    triggers = [
        _log(now - timedelta(hours=1), "rss_news"),
    ]

    fpath = tmp_path / "flags.jsonl"
    tpath = tmp_path / "triggers.jsonl"
    fpath.write_text("\n".join(_j(r) for r in flags))
    tpath.write_text("\n".join(_j(r) for r in triggers))

    out_hour = compute_origin_trends(fpath, tpath, days=1, interval="hour")
    series_hour = out_hour["origins"][0]["buckets"]
    assert len(series_hour) == 2  # two hour buckets

    out_day = compute_origin_trends(fpath, tpath, days=1, interval="day")
    series_day = out_day["origins"][0]["buckets"]
    assert len(series_day) == 1   # collapsed to one day


def test_origin_trends_api_empty_logs(tmp_path: Path, monkeypatch):
    # Point LOGS_DIR at tmp so API reads empty files
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path)

    (tmp_path / "retraining_log.jsonl").write_text("")
    (tmp_path / "retraining_triggered.jsonl").write_text("")

    client = TestClient(app)
    res = client.get("/internal/origin-trends?days=7&interval=day")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data["origins"], list)