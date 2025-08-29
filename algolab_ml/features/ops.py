from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# —— 数值型：多项式/交互项 ——
def add_polynomial(df: pd.DataFrame, cols: List[str], degree: int=2, include_bias: bool=False) -> Tuple[pd.DataFrame, List[str]]:
    sub = df[cols].astype(float)
    pf = PolynomialFeatures(degree=degree, include_bias=include_bias)
    arr = pf.fit_transform(sub.values)
    names = pf.get_feature_names_out(cols).tolist()
    out = df.copy()
    keep = []
    for i, name in enumerate(names):
        if name in cols:
            continue
        new_col = f"poly__{name}"
        out[new_col] = arr[:, i]
        keep.append(new_col)
    return out, keep

def add_interactions(df: pd.DataFrame, pairs: List[List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    created = []
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            col = f"inter__{a}_x_{b}"
            out[col] = pd.to_numeric(df[a], errors="coerce") * pd.to_numeric(df[b], errors="coerce")
            created.append(col)
    return out, created

def add_bins(df: pd.DataFrame, cols: List[str], method: str="quantile", bins: int=5, suffix: str="bin") -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    created = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if method == "quantile":
            new = pd.qcut(s, q=bins, duplicates="drop")
        else:
            new = pd.cut(s, bins=bins)
        new_col = f"{c}__{suffix}{bins}"
        out[new_col] = new.astype(str)
        created.append(new_col)
    return out, created

# —— 日期展开（惰性 + 缺失安全） ——
def add_datetime_parts(df: pd.DataFrame, mapping: Dict[str, List[str]], na_value: int = -1) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    created = []
    for col, parts in mapping.items():
        if col not in out.columns:
            continue
        s = pd.to_datetime(out[col], errors="coerce")

        def to_int(series) -> pd.Series:
            # 兼容可空整型，缺失统一填充为 -1，再转为普通 int64
            return pd.Series(series, index=out.index).astype("Int64").fillna(na_value).astype("int64")

        for p in parts:
            new_col = f"{col}__{p}"
            if p == "year":
                out[new_col] = to_int(s.dt.year)
            elif p == "month":
                out[new_col] = to_int(s.dt.month)
            elif p == "day":
                out[new_col] = to_int(s.dt.day)
            elif p == "hour":
                out[new_col] = to_int(s.dt.hour)
            elif p == "dow":
                out[new_col] = to_int(s.dt.dayofweek)
            elif p == "week":
                wk = s.dt.isocalendar().week
                out[new_col] = to_int(wk)
            elif p == "is_month_start":
                out[new_col] = s.dt.is_month_start.fillna(False).astype(int)
            elif p == "is_month_end":
                out[new_col] = s.dt.is_month_end.fillna(False).astype(int)
            else:
                # 未知关键字直接跳过
                continue
            created.append(new_col)
    return out, created

# —— 频次编码 ——
def add_frequency_encode(df: pd.DataFrame, cols: List[str], as_freq: bool=True) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    created = []
    n = len(df)
    for c in cols:
        if c not in df.columns:
            continue
        vc = df[c].astype(str).value_counts(dropna=False)
        mapping = (vc / n) if as_freq else vc
        new_col = f"{c}__{'freq' if as_freq else 'count'}"
        out[new_col] = df[c].astype(str).map(mapping).fillna(0).astype(float)
        created.append(new_col)
    return out, created

# —— 文本基础特征 ——
def add_text_basic(df: pd.DataFrame, cols: List[str], metrics: Optional[List[str]]=None) -> Tuple[pd.DataFrame, List[str]]:
    metrics = metrics or ["length", "num_alpha", "num_digit"]
    out = df.copy()
    created = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].astype(str).fillna("")
        if "length" in metrics:
            new = f"{c}__len"
            out[new] = s.str.len().astype(int); created.append(new)
        if "num_alpha" in metrics:
            new = f"{c}__alpha"
            out[new] = s.str.count(r"[A-Za-z]"); created.append(new)
        if "num_digit" in metrics:
            new = f"{c}__digit"
            out[new] = s.str.count(r"\d"); created.append(new)
        if "num_space" in metrics:
            new = f"{c}__space"
            out[new] = s.str.count(r"\s"); created.append(new)
    return out, created
