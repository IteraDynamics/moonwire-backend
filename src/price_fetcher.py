import requests
import time

# Simple in-memory cache to avoid excessive API hits
_price_cache = {}
CACHE_TTL = 300  # seconds (5 minutes)

# Basic map for normalized CoinGecko IDs
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "DOGE": "dogecoin",
    "SOL": "solana",
    "ADA": "cardano"
    # Add more as needed
}

def get_price_usd(asset: str):
    asset = asset.upper()
    coingecko_id = COINGECKO_IDS.get(asset)
    if not coingecko_id:
        print(f"[Price Fetcher] Unsupported asset: {asset}")
        return None

    # Check cache
    now = time.time()
    if asset in _price_cache:
        cached_price, timestamp = _price_cache[asset]
        if now - timestamp < CACHE_TTL:
            return cached_price

    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usd"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        price = data[coingecko_id]["usd"]

        # Update cache
        _price_cache[asset] = (price, now)
        return price

    except Exception as e:
        print(f"[Price Fetcher] Error fetching price for {asset}: {e}")
        return None

# Example use:
# print(get_price_usd("BTC"))