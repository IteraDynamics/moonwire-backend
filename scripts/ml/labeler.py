# scripts/ml/labeler.py
from __future__ import annotations
import pandas as pd

def label_next_horizon(features: pd.DataFrame, horizon_h: int = 1) -> pd.DataFrame:
    df = features.copy()
    future_close = df["close"].shift(-horizon_h)
    df["y_long"] = (future_close > df["close"]).astype(int)
    # drop last h rows without future
    df = df.iloc[:-horizon_h].reset_index(drop=True)
    return df