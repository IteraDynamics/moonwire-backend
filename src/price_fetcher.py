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

HEADERS = {
    "User-Agent": "moonwire-signal-engine/1.0"
}

def get_price_usd(asset: str):
    asset = asset.upper()
    coingecko_id = COINGECKO_IDS.get(asset)
    if not coingecko_id:
        print(f"[Price Fetcher] Unsupported asset: {asset}")
        return None

    now = time.time()
    if asset in _price_cache:
        cached_price, timestamp = _price_cache[asset]
        if now - timestamp < CACHE_TTL:
            return cached_price

    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usd"
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        data = response.json()
        price = data[coingecko_id]["usd"]
        _price_cache[asset] = (price, now)
        return price

    except Exception as e:
        print(f"[Price Fetcher] Error fetching price for {asset}: {e}")
        return None

def bulk_price_fetch(assets):
    now = time.time()
    ids_to_fetch = []
    asset_to_id = {}
    result = {}

    for asset in assets:
        asset = asset.upper()
        if asset in _price_cache:
            cached_price, timestamp = _price_cache[asset]
            if now - timestamp < CACHE_TTL:
                result[asset] = cached_price
                continue

        coingecko_id = COINGECKO_IDS.get(asset)
        if coingecko_id:
            ids_to_fetch.append(coingecko_id)
            asset_to_id[coingecko_id] = asset

    if not ids_to_fetch:
        return result

    try:
        ids_param = ",".join(ids_to_fetch)
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={ids_param}"
        time.sleep(1.5)  # Prevent burst rate issues
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        data = response.json()

        for item in data:
            cg_id = item.get("id")
            price = item.get("current_price")
            asset = asset_to_id.get(cg_id)
            if asset and price is not None:
                _price_cache[asset] = (price, now)
                result[asset] = price

    except Exception as e:
        print(f"[Price Fetcher] Bulk fetch error: {e}")

    if not result:
        print("[Price Fetcher] Using fallback mock price data")
        result = {
            "BTC": 68000,
            "ETH": 3100,
            "DOGE": 0.14,
            "SOL": 175,
            "ADA": 0.48
        }

    return result