import json
from datetime import datetime
import uuid

# Create a test signal for Standard tier
signal = {
    "id": str(uuid.uuid4()),
    "symbol": "BTC",
    "direction": "long",
    "confidence": 0.67,
    "price": 69420.00,
    "ts": datetime.utcnow().isoformat() + "Z",
    "tier": "standard"
}

# Save to standard signals file
with open('models/standard/signals.jsonl', 'w') as f:
    f.write(json.dumps(signal) + '\n')

print("✅ Test signal created!")
print(json.dumps(signal, indent=2))