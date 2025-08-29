from __future__ import annotations
import pandas as pd

def add_date_parts(df: pd.DataFrame, col: str, drop: bool=False) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[col])
    out[f"{col}_year"] = dt.dt.year
    out[f"{col}_month"] = dt.dt.month
    out[f"{col}_day"] = dt.dt.day
    out[f"{col}_dow"] = dt.dt.dayofweek
    out[f"{col}_week"] = dt.dt.isocalendar().week.astype(int)
    if drop:
        out = out.drop(columns=[col])
    return out

def add_lag_roll(df: pd.DataFrame, group_key: str, value_col: str, lags=(1,7,14), windows=(3,7)):
    out = df.copy()
    out = out.sort_values([group_key])
    for L in lags:
        out[f"{value_col}_lag{L}"] = out.groupby(group_key)[value_col].shift(L)
    for W in windows:
        out[f"{value_col}_rollmean{W}"] = out.groupby(group_key)[value_col].transform(lambda s: s.shift(1).rolling(W).mean())
        out[f"{value_col}_rollstd{W}"]  = out.groupby(group_key)[value_col].transform(lambda s: s.shift(1).rolling(W).std())
    return out
