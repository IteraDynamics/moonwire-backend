# MoonWire Signal Engine – Backend

This is the backend engine for **MoonWire** — a real-time crypto signal platform that blends social sentiment, news narratives, and market behavior into actionable indicators.

The current backend supports both **live market discovery** and **mock-mode sentiment APIs** as we prepare for model-based scoring and signal blending.

---

## Overview

MoonWire's backend is designed to ingest, analyze, and return signal-ready data for frontend display and future ML training. This includes:

- Real-time sleeper signal scanning (market-based)
- Twitter + news sentiment APIs (mock or cached)
- Autonomous scoring loop (10-minute interval)
- API endpoints used by the MoonWire frontend
- Internal-only experimental modules for scoring logic

---

## Key Features

- **FastAPI server** — lightweight, async, auto-launch on deploy
- **Live CoinGecko ingestion** — top 250 coins pulled for signal analysis
- **Sleeper detection engine** — identifies unexpected price/volume anomalies
- **Mock sentiment endpoints** — support frontend beta with fallback data
- **Auto-loop analysis engine** — ingestion → analysis → dispatch every 10 minutes
- **Lightweight local cache** — Redis simulated via in-memory storage
- **Modular design** — core logic isolated in `src/` for easy upgrades

---

## Tech Stack

- **Python + FastAPI** — modern async API framework
- **Uvicorn** — ASGI production-ready server
- **Requests** — live market data ingestion
- **Render** — serverless backend hosting
- **GitHub** — version control + CI/CD triggers

---

## Core Modules

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI app entrypoint, includes sentiment/news routers |
| `src/auto_loop.py` | Background thread running signal scan every 10 minutes |
| `src/ingest_discovery.py` | Pulls top 250 assets from CoinGecko |
| `src/signal_generator.py` | Detects sleeper signals via price/volume anomalies |
| `src/news_router.py` | News sentiment score API (mock or cached) |
| `src/sentiment_news.py` | Sentiment analysis logic for headlines |
| `src/cache.py` | Lightweight dict-based cache (MVP-safe) |
| `src/logger.py` | Structured log entries for tracking system activity |

---

## API Endpoints

| Route | Purpose |
|-------|---------|
| `/sentiment/twitter` | Returns mock sentiment data for selected asset |
| `/news-sentiment` | Returns mock or cached news sentiment scores |
| `/signals/sleepers` | Internal route (future public dashboard feed) |

---

## How It Works

1. Server boots via Render → launches FastAPI app
2. Background loop runs every 10 minutes:
   - Pulls top assets from CoinGecko
   - Evaluates for price/volume spikes
   - Logs matched signals
3. Frontend queries mock API endpoints for now
4. Ready for swap to real scoring model as backend evolves

---

## Deployment Notes

- Hosted on **Render.com** with auto-loop enabled
- No DB or Redis required — runs on local memory and simple caching
- Rate-limit aware — all ingestion paced for free API tiers
- Logs print to console for early-stage monitoring

---

## In Progress

- Model-based sentiment scoring (social + news)
- Composite signal engine (score, label, drivers, trend)
- Event tracking and confidence metadata
- Feedback and training data export
- Private signal testing dashboard

---

## Mission

> **Move faster than the crowd. Catch what others miss.  
> MoonWire.  
> Built for signal hunters.**

---