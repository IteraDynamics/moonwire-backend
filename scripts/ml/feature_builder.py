# scripts/ml/feature_builder.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

def _features_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1h"] = df["close"].pct_change(1)
    df["r_1h"] = df["ret_1h"]
    df["r_3h"] = df["close"].pct_change(3)
    df["r_6h"] = df["close"].pct_change(6)

    # volatility
    df["vol_6h"] = df["ret_1h"].rolling(6).std()
    # ATR approx over 14h (true range on OHLC)
    tr = (df["high"] - df["low"]).abs()
    tr_prev_close = (df["high"] - df["close"].shift(1)).abs().combine(
        (df["low"] - df["close"].shift(1)).abs(), max
    )
    df["atr_14h"] = pd.concat([tr, tr_prev_close], axis=1).max(axis=1).rolling(14).mean()

    # momentum
    df["sma_6h"] = df["close"].rolling(6).mean()
    df["sma_24h"] = df["close"].rolling(24).mean()
    df["sma_gap"] = df["sma_6h"]/df["sma_24h"] - 1.0

    # regime flag
    thresh = df["vol_6h"].quantile(0.75)
    df["high_vol"] = (df["vol_6h"] > thresh).astype(int)

    # social proxies (neutral default)
    df["social_score"] = 0.5

    # clean
    df = df.dropna().reset_index(drop=True)
    return df

def build_features(df_prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out = {}
    for sym, df in df_prices.items():
        f = _features_for_symbol(df)
        out[sym] = f
    return out