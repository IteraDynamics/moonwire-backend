import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.analytics.source_yield import compute_source_yield


def make_log(ts: datetime, origin: str, trigger=False):
    log = {"timestamp": ts.isoformat()}
    if trigger:
        log["meta"] = {"origin": origin}
    else:
        log["origin"] = origin
    return log


def test_yield_plan_happy_path(tmp_path: Path):
    now = datetime.now(timezone.utc)
    flags = [
        make_log(now, "twitter"),
        make_log(now, "twitter"),
        make_log(now, "twitter"),
        make_log(now, "reddit"),
        make_log(now, "reddit"),
    ]
    triggers = [
        make_log(now, "twitter", trigger=True),
        make_log(now, "reddit", trigger=True)
    ]

    flags_path = tmp_path / "flags.jsonl"
    triggers_path = tmp_path / "triggers.jsonl"
    flags_path.write_text("\n".join(json.dumps(f) for f in flags))
    triggers_path.write_text("\n".join(json.dumps(t) for t in triggers))

    result = compute_source_yield(flags_path, triggers_path, days=7, min_events=1, alpha=0.5)

    assert result["totals"]["flags"] == 5
    assert result["totals"]["triggers"] == 2
    assert len(result["budget_plan"]) == 2
    assert abs(sum(x["pct"] for x in result["budget_plan"]) - 100) < 0.5


def test_yield_plan_min_events(tmp_path: Path):
    now = datetime.now(timezone.utc)
    flags = [make_log(now, "rss_news")]  # only 1 event
    flags_path = tmp_path / "flags.jsonl"
    flags_path.write_text("\n".join(json.dumps(f) for f in flags))

    result = compute_source_yield(flags_path, Path("missing.jsonl"), days=7, min_events=5, alpha=0.5)

    assert result["origins"][0]["origin"] == "rss_news"
    assert result["origins"][0]["eligible"] is False
    assert result["budget_plan"] == []


def test_yield_plan_alpha_extremes(tmp_path: Path):
    now = datetime.now(timezone.utc)
    flags = [make_log(now, "a")] * 10 + [make_log(now, "b")] * 2
    triggers = [make_log(now, "a", trigger=True), make_log(now, "b", trigger=True)]

    flags_path = tmp_path / "flags.jsonl"
    triggers_path = tmp_path / "triggers.jsonl"
    flags_path.write_text("\n".join(json.dumps(f) for f in flags))
    triggers_path.write_text("\n".join(json.dumps(t) for t in triggers))

    result_vol = compute_source_yield(flags_path, triggers_path, 7, 1, alpha=0.0)
    result_conv = compute_source_yield(flags_path, triggers_path, 7, 1, alpha=1.0)

    top_vol = result_vol["budget_plan"][0]["origin"]
    top_conv = result_conv["budget_plan"][0]["origin"]

    assert top_vol == "a"
    assert top_conv == "b"


def test_missing_triggers(tmp_path: Path):
    now = datetime.now(timezone.utc)
    flags = [make_log(now, "market_feed"), make_log(now, "market_feed")]
    flags_path = tmp_path / "flags.jsonl"
    flags_path.write_text("\n".join(json.dumps(f) for f in flags))

    result = compute_source_yield(flags_path, Path("does_not_exist.jsonl"), 7, 1, alpha=0.5)
    assert result["origins"][0]["trigger_rate"] == 0.0
    assert result["budget_plan"][0]["origin"] == "market_feed"


def test_window_filter(tmp_path: Path):
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=10)
    flags = [make_log(old, "twitter"), make_log(old, "reddit")]
    triggers = [make_log(old, "twitter", trigger=True)]

    flags_path = tmp_path / "flags.jsonl"
    triggers_path = tmp_path / "triggers.jsonl"
    flags_path.write_text("\n".join(json.dumps(f) for f in flags))
    triggers_path.write_text("\n".join(json.dumps(t) for t in triggers))

    result = compute_source_yield(flags_path, triggers_path, days=5, min_events=1, alpha=0.7)
    assert result["totals"]["flags"] == 0
    assert result["budget_plan"] == []