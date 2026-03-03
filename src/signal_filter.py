# src/signal_filter.py

from datetime import datetime

# Store recent alert timestamps per asset
_recent_alerts = {}

def is_signal_valid(signal):
    asset = signal['asset']
    movement = signal['price_change']
    volume = signal['volume']
    now = signal['timestamp']

    MIN_MOVEMENT = 2.0  # Temporarily lowered for testing (normal: 7.0)
    MIN_VOLUME = 10_000_000
    COOLDOWN_MINUTES = 30

    if abs(movement) < MIN_MOVEMENT:  # Use absolute value to catch both positive and negative movements
        return False

    if volume < MIN_VOLUME:
        return False

    if asset in _recent_alerts:
        last_time, last_movement = _recent_alerts[asset]
        elapsed = (now - last_time).total_seconds() / 60
        if elapsed < COOLDOWN_MINUTES and abs(movement - last_movement) < 2:
            return False

    _recent_alerts[asset] = (now, movement)
    return True
