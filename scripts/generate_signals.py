"""Generate signals from trained models"""
import sys
from pathlib import Path

# CRITICAL: Add parent directory to path so 'scripts' module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
import uuid
import joblib
import numpy as np
import requests

# Paths
ROOT = Path(__file__).parent.parent
STANDARD_MODELS = ROOT / "models" / "standard"
ELITE_MODELS = ROOT / "models" / "elite"

def get_current_price(symbol):
    """Fetch current price from CoinGecko"""
    coin_ids = {"BTC": "bitcoin", "ETH": "ethereum"}
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": coin_ids[symbol], "vs_currencies": "usd"}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return data[coin_ids[symbol]]["usd"]
    except Exception as e:
        print(f"  ⚠️  Could not fetch {symbol} price: {e}")
        return 0.0

def generate_dummy_features():
    """Generate dummy features (8 features for your model)"""
    return np.random.randn(1, 8)

def generate_signals(tier, symbols):
    """Generate signals for a specific tier"""
    
    model_dir = STANDARD_MODELS if tier == "standard" else ELITE_MODELS
    signals_file = model_dir / "signals.jsonl"
    
    print(f"\n{'='*50}")
    print(f"Generating {tier.upper()} tier signals")
    print(f"{'='*50}")
    
    for symbol in symbols:
        try:
            # Load model
            model_path = model_dir / f"{symbol}_model.joblib"
            if not model_path.exists():
                print(f"  ❌ {symbol}: Model not found")
                continue
            
            model = joblib.load(model_path)
            
            # Generate dummy features
            X = generate_dummy_features()
            
            # Get prediction
            prediction = model.predict_proba(X)[0][1]
            
            # Determine direction
            direction = "long" if prediction > 0.5 else "short"
            confidence = prediction if direction == "long" else (1 - prediction)
            
            # Skip if below threshold
            if confidence < 0.55:
                print(f"  {symbol}: Below threshold ({confidence:.2%}), skipping")
                continue
            
            # Get price
            price = get_current_price(symbol)
            
            # Create signal
            signal = {
                "id": str(uuid.uuid4()),
                "symbol": symbol,
                "direction": direction,
                "confidence": float(confidence),
                "price": price,
                "ts": datetime.utcnow().isoformat() + "Z",
                "tier": tier
            }
            
            # Save
            with open(signals_file, 'a') as f:
                f.write(json.dumps(signal) + '\n')
            
            print(f"  ✅ {symbol}: {direction.upper()} @ {confidence:.2%}, ${price:,.2f}")
            
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")
    
    print(f"{'='*50}\n")

if __name__ == "__main__":
    import sys
    
    tier = sys.argv[1] if len(sys.argv) > 1 else "standard"
    symbols = ["BTC", "ETH"]
    
    if tier == "both":
        generate_signals("standard", symbols)
        generate_signals("elite", symbols)
    else:
        generate_signals(tier, symbols)