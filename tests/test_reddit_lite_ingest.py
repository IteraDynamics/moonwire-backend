# tests/test_reddit_lite_ingest.py
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import importlib

import pytest

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

RSS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>r/CryptoCurrency new posts</title>
  <entry>
    <title>ETF inflows spike</title>
    <id>tag:reddit.com,2005:/r/CryptoCurrency/comments/xyz123</id>
    <updated>""" + _iso(datetime.now(timezone.utc) - timedelta(hours=1)) + """</updated>
    <author><name>alice</name></author>
    <link rel="alternate" href="https://www.reddit.com/r/CryptoCurrency/comments/xyz123/etf_inflows_spike/"/>
  </entry>
  <entry>
    <title>Spot approvals chatter</title>
    <id>tag:reddit.com,2005:/r/CryptoCurrency/comments/abc987</id>
    <updated>""" + _iso(datetime.now(timezone.utc) - timedelta(hours=2)) + """</updated>
    <author><name>bob</name></author>
    <link rel="alternate" href="https://www.reddit.com/r/CryptoCurrency/comments/abc987/spot_approvals_chatter/"/>
  </entry>
</feed>
"""

class DummyResp:
    def __init__(self, code=200, text="{}", json_payload=None):
        self.status_code = code
        self.text = text
        self._json = json_payload

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        if self._json is not None:
            return self._json
        return json.loads(self.text or "{}")


def test_rss_mode_ingest(monkeypatch, tmp_path):
    # env: rss mode, 24h lookback, single sub
    monkeypatch.setenv("MW_REDDIT_MODE", "rss")
    monkeypatch.setenv("MW_REDDIT_SUBS", "CryptoCurrency")
    monkeypatch.setenv("MW_REDDIT_SORT", "new")
    monkeypatch.setenv("MW_REDDIT_LOOKBACK_H", "24")
    monkeypatch.setenv("DEMO_MODE", "false")

    # mock requests.Session.request to return our RSS
    import scripts.social.reddit_client_lite as rc
    importlib.reload(rc)

    def fake_request(self, method, url, timeout=None, **kwargs):
        assert "reddit.com" in url
        return DummyResp(200, RSS_SAMPLE, None)

    monkeypatch.setattr(rc.requests.Session, "request", fake_request, raising=True)

    # run ingest
    import scripts.social.reddit_lite_ingest as ing
    importlib.reload(ing)

    paths = ing.IngestPaths(
        logs_dir=tmp_path / "logs",
        models_dir=tmp_path / "models",
        artifacts_dir=tmp_path / "artifacts",
    )
    out = ing.run_ingest(paths=paths)

    # artifact exists
    j = json.loads((paths.models_dir / "social_reddit_context.json").read_text())
    assert j.get("mode") == "rss"
    assert j.get("window_hours") == 24
    assert "CryptoCurrency" in j.get("subs", [])
    # counts present
    c = j.get("counts", {}).get("CryptoCurrency", {})
    assert c.get("posts", 0) >= 2

    # log lines exist
    logf = paths.logs_dir / "social_reddit.jsonl"
    assert logf.exists()
    lines = [ln for ln in logf.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 2

    # pngs exist
    p1 = paths.artifacts_dir / "reddit_activity_CryptoCurrency.png"
    p2 = paths.artifacts_dir / "reddit_bursts_CryptoCurrency.png"
    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0


def test_demo_mode_deterministic(monkeypatch, tmp_path):
    monkeypatch.setenv("MW_REDDIT_MODE", "rss")  # doesn't matter in demo
    monkeypatch.setenv("MW_REDDIT_SUBS", "Bitcoin,Solana")
    monkeypatch.setenv("MW_REDDIT_LOOKBACK_H", "12")
    monkeypatch.setenv("MW_DEMO", "true")

    import scripts.social.reddit_lite_ingest as ing
    importlib.reload(ing)
    out = ing.run_ingest(paths=ing.IngestPaths(
        logs_dir=tmp_path / "logs",
        models_dir=tmp_path / "models",
        artifacts_dir=tmp_path / "artifacts",
    ))

    j = json.loads((tmp_path / "models" / "social_reddit_context.json").read_text())
    assert j.get("demo") is True
    assert set(j.get("subs", [])) == {"Bitcoin", "Solana"}
    # Should have counts for both
    cnt = j.get("counts", {})
    assert "Bitcoin" in cnt and "Solana" in cnt
    # ensure PNGs created
    for sub in ("Bitcoin", "Solana"):
        assert (tmp_path / "artifacts" / f"reddit_activity_{sub}.png").exists()
        assert (tmp_path / "artifacts" / f"reddit_bursts_{sub}.png").exists()


def test_summary_section_integration(monkeypatch, tmp_path):
    # use demo to avoid network
    monkeypatch.setenv("MW_REDDIT_SUBS", "ethtrader")
    monkeypatch.setenv("MW_REDDIT_LOOKBACK_H", "6")
    monkeypatch.setenv("MW_DEMO", "true")

    # minimal SummaryContext stand-in
    class Ctx:
        def __init__(self, logs_dir, models_dir, artifacts_dir, is_demo):
            self.logs_dir = logs_dir
            self.models_dir = models_dir
            self.artifacts_dir = artifacts_dir
            self.is_demo = is_demo

    ctx = Ctx(tmp_path / "logs", tmp_path / "models", tmp_path / "artifacts", True)

    import scripts.summary_sections.social_context_reddit as scr
    importlib.reload(scr)

    md: List[str] = []
    scr.append(md, ctx)
    out = "\n".join(md)
    assert "Social Context — Reddit" in out
    assert "ethtrader" in out

    j = json.loads((tmp_path / "models" / "social_reddit_context.json").read_text())
    assert j.get("demo") is True
    assert "ethtrader" in j.get("subs", [])