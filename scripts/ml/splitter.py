# scripts/ml/splitter.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterator, Tuple

def walk_forward_splits(df: pd.DataFrame, n_splits: int = 3,
                        train_days: int = 60, test_days: int = 30) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    df = df.sort_values("ts").reset_index(drop=True)
    # assume hourly bars
    train_len = train_days * 24
    test_len = test_days * 24
    total = len(df)
    starts = []
    i = 0
    # Build rolling windows from the back so we always have full windows
    while True:
        end = i + train_len + test_len
        if end > total:
            break
        starts.append(i)
        i += test_len // max(n_splits, 1)  # stride forward by chunk of test window
        if len(starts) >= n_splits:
            break
    if not starts:
        # fall back to single split if dataset short
        starts = [max(0, total - (train_len + test_len))]

    for s in starts:
        train_idx = np.arange(s, s + train_len)
        test_idx = np.arange(s + train_len, s + train_len + test_len)
        yield train_idx, test_idx