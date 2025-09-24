import json, importlib, os
from pathlib import Path
from datetime import datetime, timezone, timedelta

from scripts.summary_sections.common import SummaryContext, _iso


def _write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_headers_and_backoff(monkeypatch, tmp_path):
    # mock Session.request to verify header + simulate 429 then 200
    calls = {"n": 0}
    def fake_request(self, method, url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        # header check when key present
        assert headers.get("x-cg-pro-api-key") == "TESTKEY"
        class R:
            def __init__(self, code, payload):
                self.status_code = code
                self._payload = payload
                self.text = json.dumps(payload)
            def json(self): return self._payload
        if calls["n"] == 1:
            return R(429, {"error":"rate limit"})
        # endpoints used by ingest: simple/price and market_chart
        if "simple/price" in url:
            return R(200, {"bitcoin": {"usd": 60000.0}})
        if "market_chart" in url:
            now = datetime.now(timezone.utc)
            return R(200, {"prices": [
                [(now - timedelta(hours=2)).timestamp()*1000, 59000.0],
                [(now - timedelta(hours=1)).timestamp()*1000, 60000.0],
                [now.timestamp()*1000, 61000.0],
            ]})
        return R(200, {})
    import requests
    monkeypatch.setenv("MW_CG_API_KEY", "TESTKEY")
    monkeypatch.setenv("MW_DEMO", "false")
    monkeypatch.setenv("MW_CG_COINS", "bitcoin")
    monkeypatch.setenv("MW_CG_LOOKBACK_H", "2")

    from scripts.market import coingecko_client as cg
    importlib.reload(cg)
    monkeypatch.setattr(cg.requests.Session, "request", fake_request, raising=True)

    client = cg.CoinGeckoClient()
    # first call will 429, second returns
    data = client.simple_price(["bitcoin"], "usd", True)
    assert "bitcoin" in data
    chart = client.market_chart_days("bitcoin", "usd", 1)
    assert "prices" in chart


def test_ingest_demo_and_artifacts(monkeypatch, tmp_path):
    # Force demo
    monkeypatch.setenv("MW_DEMO", "true")
    monkeypatch.setenv("MW_CG_COINS", "bitcoin,ethereum,solana")
    monkeypatch.setenv("MW_CG_LOOKBACK_H", "24")
    models = tmp_path / "models"; models.mkdir(parents=True, exist_ok=True)
    logs = tmp_path / "logs"; logs.mkdir(parents=True, exist_ok=True)
    arts = tmp_path / "artifacts"; arts.mkdir(parents=True, exist_ok=True)

    from scripts.market.ingest_market import run_ingest
    out = run_ingest(logs, models, arts)

    # JSON artifact
    jpath = models / "market_context.json"
    assert jpath.exists()
    data = json.loads(jpath.read_text())
    assert data.get("demo") is True
    assert set(data.get("coins", [])) >= {"bitcoin","ethereum","solana"}
    for coin in data["coins"]:
        assert len(data["series"][coin]) >= 12  # at least 12 hourly points

    # PNGs exist
    for c in ("bitcoin","ethereum","solana"):
        p1 = arts / f"market_trend_price_{c}.png"
        p2 = arts / f"market_trend_returns_{c}.png"
        assert p1.exists() and p1.stat().st_size > 0
        assert p2.exists() and p2.stat().st_size > 0

    # Log line appended
    logf = logs / "market_prices.jsonl"
    assert logf.exists()
    lines = [ln for ln in logf.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 3  # one per coin per run


def test_market_context_summary_integration(monkeypatch, tmp_path):
    # Ensure summary section writes markdown and uses ingest
    monkeypatch.setenv("MW_DEMO", "true")
    models = tmp_path / "models"; models.mkdir(parents=True, exist_ok=True)
    logs = tmp_path / "logs"; logs.mkdir(parents=True, exist_ok=True)

    from scripts.summary_sections.common import SummaryContext
    from scripts.summary_sections import market_context
    ctx = SummaryContext(logs_dir=logs, models_dir=models, is_demo=True)

    md=[]
    market_context.append(md, ctx)
    txt = "\n".join(md)
    assert "Market Context (CoinGecko" in txt
    assert "Data via CoinGecko API" in txt