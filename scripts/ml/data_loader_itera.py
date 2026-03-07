# scripts/ml/data_loader_itera.py
"""
Custom data loader for Itera historical CSV files.
Reads from /home/clawd/clawd/iteradynamics/data/

File format (Coinbase/Itera):
    Timestamp,Open,High,Low,Close,Volume
    2019-01-01 00:00:00,3747.39,3759.98,3741.44,3750.91,1451.38
    ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


# ---- Paths -----------------------------------------------------------------

ITERA_DATA_DIR = Path("/home/clawd/clawd/iteradynamics/data")

# Symbol to file mapping
SYMBOL_FILES = {
    "BTC": "btcusd_3600s_2019-01-01_to_2025-12-30.csv",
    "ETH": "ethusd_3600s_2019-01-01_to_2025-12-30.csv",
}


# ---- Public API ------------------------------------------------------------

def load_prices_itera(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load hourly OHLCV for the requested symbols from Itera CSV files.
    
    Args:
        symbols: List of symbol codes (e.g., ['BTC', 'ETH'])
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    
    Returns:
        dict[symbol] -> DataFrame with columns:
            ts (UTC tz-aware), open, high, low, close, volume
    """
    out: Dict[str, pd.DataFrame] = {}
    
    for symbol in symbols:
        symbol = symbol.upper()
        
        if symbol not in SYMBOL_FILES:
            raise ValueError(
                f"Symbol {symbol} not found. Available: {list(SYMBOL_FILES.keys())}"
            )
        
        file_path = ITERA_DATA_DIR / SYMBOL_FILES[symbol]
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Make sure to download historical data first."
            )
        
        print(f"Loading {symbol} from {file_path.name}...")
        df = _load_csv(file_path)
        
        # Apply date filters if provided
        if start_date:
            start_ts = pd.to_datetime(start_date, utc=True)
            df = df[df["ts"] >= start_ts]
        
        if end_date:
            end_ts = pd.to_datetime(end_date, utc=True)
            df = df[df["ts"] <= end_ts]
        
        df = _finalize_schema(df)
        
        print(f"  Loaded {len(df)} bars from {df['ts'].min()} to {df['ts'].max()}")
        
        out[symbol] = df
    
    return out


# ---- CSV Loading -----------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    """
    Load Itera CSV format:
        Timestamp,Open,High,Low,Close,Volume
        2019-01-01 00:00:00,3747.39,3759.98,3741.44,3750.91,1451.38
    
    Returns DataFrame with standardized lowercase column names.
    """
    df = pd.read_csv(path)
    
    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    
    # Rename 'timestamp' to 'ts' for consistency with MoonWire
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    
    # Parse timestamp as UTC
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    
    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Sort by timestamp
    df = df.sort_values("ts").reset_index(drop=True)
    
    return df


# ---- Post-processing -------------------------------------------------------

def _finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure schema matches MoonWire expectations:
        - ts: UTC-aware timestamp
        - open, high, low, close, volume: float
        - No missing values in critical columns
    """
    cols = ["ts", "open", "high", "low", "close", "volume"]
    
    df = df.copy()
    
    # Ensure all columns exist
    for c in cols:
        if c not in df.columns:
            if c == "volume":
                df[c] = 0.0
            else:
                df[c] = np.nan
    
    # Filter to required columns
    df = df[cols]
    
    # Drop rows with missing critical data
    df = df.dropna(subset=["ts", "close"])
    
    # Fill missing OHLC from close
    for col in ["open", "high", "low"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df["close"])
    
    # Fill missing volume with 0
    df["volume"] = df["volume"].fillna(0.0)
    
    # Final sort and reset index
    df = df.sort_values("ts").reset_index(drop=True)
    
    # Validation
    assert df["ts"].is_monotonic_increasing, "Timestamps must be monotonically increasing"
    assert not df["close"].isna().any(), "Close prices must not have NaN values"
    
    return df


# ---- Stats Helper ----------------------------------------------------------

def get_data_stats(symbol: str) -> Dict:
    """Get statistics about a symbol's data file."""
    if symbol.upper() not in SYMBOL_FILES:
        return {"error": f"Symbol {symbol} not in SYMBOL_FILES"}
    
    file_path = ITERA_DATA_DIR / SYMBOL_FILES[symbol.upper()]
    
    if not file_path.exists():
        return {"error": f"File not found: {file_path}"}
    
    df = _load_csv(file_path)
    
    return {
        "symbol": symbol.upper(),
        "file": file_path.name,
        "bars": len(df),
        "start": str(df["ts"].min()),
        "end": str(df["ts"].max()),
        "days": (df["ts"].max() - df["ts"].min()).days,
        "price_range": f"${df['close'].min():.2f} - ${df['close'].max():.2f}",
        "completeness": f"{(1 - df['close'].isna().sum() / len(df)) * 100:.1f}%",
    }


# ---- CLI Test --------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        symbols = [s.strip().upper() for s in sys.argv[1:]]
    else:
        symbols = ["BTC", "ETH"]
    
    print("\n=== Itera Data Loader Test ===\n")
    
    for sym in symbols:
        stats = get_data_stats(sym)
        if "error" in stats:
            print(f"❌ {sym}: {stats['error']}")
        else:
            print(f"✓ {stats['symbol']}")
            print(f"  File: {stats['file']}")
            print(f"  Bars: {stats['bars']:,}")
            print(f"  Range: {stats['start']} → {stats['end']} ({stats['days']} days)")
            print(f"  Price: {stats['price_range']}")
            print(f"  Complete: {stats['completeness']}")
            print()
    
    # Test load
    print("Loading full datasets...")
    try:
        data = load_prices_itera(symbols)
        print(f"\n✅ Successfully loaded {len(data)} symbols")
        for sym, df in data.items():
            print(f"  {sym}: {len(df)} bars")
    except Exception as e:
        print(f"\n❌ Load failed: {e}")
