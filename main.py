from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
from pathlib import Path
from typing import List, Optional
from threading import Thread

app = FastAPI(
    title="MoonWire API",
    version="1.0.0",
    description="Crypto signal intelligence API"
)

# Start auto-loop on startup
@app.on_event("startup")
def startup_event():
    from src.auto_loop import auto_loop
    loop_thread = Thread(target=auto_loop, args=(600,), daemon=True)
    loop_thread.start()
    print("🔄 Signal generation auto-loop started (every 10 minutes)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SIGNALS_FILE = Path("logs/signal_history.jsonl")

def read_signals() -> List[dict]:
    """Read all signals from JSONL file"""
    if not SIGNALS_FILE.exists():
        return []
    
    signals = []
    with open(SIGNALS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    signals.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return signals

@app.get("/")
def root():
    """API health check"""
    return {
        "name": "MoonWire API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "latest_signal": "/api/signal/latest",
            "signal_history": "/api/signal/history",
            "statistics": "/api/signal/stats"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    signals = read_signals()
    return {
        "status": "healthy",
        "signals_count": len(signals),
        "last_updated": signals[-1]["ts"] if signals else None
    }

@app.get("/api/signal/latest")
def get_latest_signal():
    """Get the most recent signal"""
    signals = read_signals()
    
    if not signals:
        raise HTTPException(
            status_code=404,
            detail="No signals generated yet. Check back soon!"
        )
    
    return signals[-1]

@app.get("/api/signal/history")
def get_signal_history(limit: int = 50, skip: int = 0):
    """
    Get recent signal history
    
    - **limit**: Number of signals to return (default 50, max 200)
    - **skip**: Number of signals to skip (for pagination)
    """
    if limit > 200:
        limit = 200
    
    signals = read_signals()
    
    # Apply pagination
    start_idx = max(0, len(signals) - skip - limit)
    end_idx = len(signals) - skip if skip > 0 else len(signals)
    
    paginated = signals[start_idx:end_idx]
    
    return {
        "count": len(paginated),
        "total": len(signals),
        "signals": paginated
    }

@app.get("/api/signal/stats")
def get_signal_stats():
    """Get performance statistics for all signals"""
    signals = read_signals()
    
    if not signals:
        return {
            "total_signals": 0,
            "realized": 0,
            "pending": 0,
            "win_rate": 0.0,
            "avg_outcome": 0.0,
            "latest_signal": None
        }
    
    # Separate realized vs pending
    realized = [s for s in signals if s.get("outcome") is not None]
    pending = [s for s in signals if s.get("outcome") is None]
    
    # Calculate wins
    wins = [s for s in realized if s.get("outcome", 0) > 0]
    losses = [s for s in realized if s.get("outcome", 0) <= 0]
    
    # Calculate averages
    win_rate = len(wins) / len(realized) if realized else 0.0
    avg_outcome = sum(s["outcome"] for s in realized) / len(realized) if realized else 0.0
    
    # Get best and worst
    best_signal = max(realized, key=lambda s: s["outcome"]) if realized else None
    worst_signal = min(realized, key=lambda s: s["outcome"]) if realized else None
    
    return {
        "total_signals": len(signals),
        "realized": len(realized),
        "pending": len(pending),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 3),
        "avg_outcome": round(avg_outcome, 4),
        "best_trade": {
            "symbol": best_signal["symbol"],
            "outcome": round(best_signal["outcome"], 4)
        } if best_signal else None,
        "worst_trade": {
            "symbol": worst_signal["symbol"],
            "outcome": round(worst_signal["outcome"], 4)
        } if worst_signal else None,
        "latest_signal": signals[-1]
    }

@app.post("/api/signal/generate")
def trigger_signal_generation():
    """Manually trigger signal generation (testing only)"""
    try:
        from src.signal_generator import generate_signals
        generate_signals()
        return {
            "status": "success",
            "message": "Signal generation triggered. Check Discord in 1-2 minutes."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    from threading import Thread
    from src.auto_loop import auto_loop
    
    # Start auto_loop in background thread
    loop_thread = Thread(target=auto_loop, args=(600,), daemon=True)
    loop_thread.start()
    print("🔄 Signal generation loop started (every 10 minutes)")
    
    print("🚀 Starting MoonWire API...")
    print("📡 API will be available at: http://localhost:8000")
    print("📚 Docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
