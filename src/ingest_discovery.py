# src/ingest_discovery.py

import requests
from datetime import datetime

# Correct CoinPaprika asset IDs
ASSET_MAP = {
    'BTC': 'btc-bitcoin',
    'ETH': 'eth-ethereum',
    'SOL': 'sol-solana'
}

def fetch_from_coinpaprika(asset_id):
    try:
        url = f"https://api.coinpaprika.com/v1/tickers/{asset_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'price_change_24h': data['quotes']['USD']['percent_change_24h'],
                'volume_now': data['quotes']['USD']['volume_24h'],
                'current_price': data['quotes']['USD']['price']
            }
    except Exception as e:
        print(f"CoinPaprika failed for {asset_id}: {e}")
    return None

def fetch_from_coingecko(asset_slug):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{asset_slug.lower()}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'price_change_24h': data['market_data']['price_change_percentage_24h'],
                'volume_now': data['market_data']['total_volume']['usd'],
                'current_price': data['market_data']['current_price']['usd']
            }
    except Exception as e:
        print(f"CoinGecko failed for {asset_slug}: {e}")
    return None

def ingest_market_data(cache):
    print(f"[{datetime.utcnow()}] Ingesting market data...")

    for symbol, paprika_id in ASSET_MAP.items():
        data = fetch_from_coinpaprika(paprika_id)
        if not data:
            print(f"Falling back to CoinGecko for {symbol}")
            data = fetch_from_coingecko(symbol.lower())

        if data:
            cache.set_signal(symbol, data)
            print(f"Cached data for {symbol}: {data}")
        else:
            print(f"Failed to fetch data for {symbol}")
