from __future__ import annotations
from typing import Optional, Tuple, Sequence, Union
import pandas as pd
from pathlib import Path

def load_csv(path: Union[str, Path], dtype: Optional[dict]=None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=dtype)

def train_test_split_df(df: pd.DataFrame, target: str, test_size: float=0.2, random_state: int=42):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target])
    y = df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique()<50 else None)
    return Xtr, Xte, ytr, yte

def detect_problem_type(y) -> str:
    import numpy as np
    n_unique = len(pd.Series(y).unique())
    if pd.api.types.is_numeric_dtype(y) and n_unique > 20:
        return "regression"
    return "classification"
