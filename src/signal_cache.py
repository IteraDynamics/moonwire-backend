# src/signal_cache.py

# Local in-memory cache for the latest signal
_latest_signal = None

def save_latest_signal(signal: dict):
    global _latest_signal
    _latest_signal = signal
    print("[Signal Cache] Signal saved.")

def get_latest_signal() -> dict:
    if _latest_signal:
        print("[Signal Cache] Returning latest signal.")
    else:
        print("[Signal Cache] No signal cached yet.")
    return _latest_signal