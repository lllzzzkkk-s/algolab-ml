from __future__ import annotations
from typing import List, Optional, Dict, Any, Iterable
import pandas as pd
import numpy as np

# ---------- 基础清洗 ----------
def basic_clean(df: pd.DataFrame, drop_cols: Optional[List[str]] = None) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.drop_duplicates()
    if drop_cols:
        keep = [c for c in out.columns if c not in set(drop_cols)]
        out = out[keep]
    return out

def fill_na(df: pd.DataFrame, num_strategy: str = "median", cat_strategy: str = "most_frequent") -> pd.DataFrame:
    from sklearn.impute import SimpleImputer
    out = df.copy()
    num_cols = out.select_dtypes(include=["number"]).columns
    cat_cols = [c for c in out.columns if c not in num_cols]
    if len(num_cols) > 0:
        out[num_cols] = SimpleImputer(strategy=num_strategy).fit_transform(out[num_cols])
    if len(cat_cols) > 0:
        out[cat_cols] = SimpleImputer(strategy=cat_strategy).fit_transform(out[cat_cols])
    return out

# ---------- 进阶清洗 ----------
def enforce_schema(df: pd.DataFrame, dtypes: Optional[Dict[str, str]] = None,
                   rename: Optional[Dict[str, str]] = None,
                   required: Optional[Iterable[str]] = None) -> pd.DataFrame:
    out = df.copy()
    if rename:
        out = out.rename(columns=rename)
    if dtypes:
        for col, dt in dtypes.items():
            if col in out.columns:
                try:
                    if dt == "category":
                        out[col] = out[col].astype("category")
                    elif dt == "datetime":
                        out[col] = pd.to_datetime(out[col], errors="coerce")
                    else:
                        out[col] = out[col].astype(dt)
                except Exception:
                    pass
    if required:
        miss = [c for c in required if c not in out.columns]
        if miss:
            raise ValueError(f"Missing required columns: {miss}. Available: {list(out.columns)}")
    return out

def drop_constant_cols(df: pd.DataFrame, threshold_unique: int = 1) -> pd.DataFrame:
    out = df.copy()
    to_drop = [c for c in out.columns if out[c].nunique(dropna=True) <= threshold_unique]
    return out.drop(columns=to_drop) if to_drop else out

def clip_outliers(df: pd.DataFrame, cols: Optional[List[str]] = None,
                  method: str = "iqr", z_thresh: float = 3.0, iqr_k: float = 1.5) -> pd.DataFrame:
    out = df.copy()
    if cols is None:
        cols = out.select_dtypes(include=["number"]).columns.tolist()
    for c in cols:
        s = out[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if method == "z":
            m, sd = s.mean(), s.std(ddof=0)
            if sd and not np.isnan(sd):
                out[c] = s.clip(m - z_thresh * sd, m + z_thresh * sd)
        else:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
            out[c] = s.clip(lo, hi)
    return out

def bucket_rare_categories(df: pd.DataFrame, cols: List[str], min_freq: int = 10, other_label: str = "_OTHER") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        vc = out[c].astype(str).value_counts(dropna=False)
        rare = set(vc[vc < min_freq].index.tolist())
        out[c] = out[c].astype(str).apply(lambda x: other_label if x in rare else x)
    return out

def parse_dates(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, fmt in mapping.items():
        if col not in out.columns:
            continue
        if fmt == "auto":
            out[col] = pd.to_datetime(out[col], errors="coerce")
        else:
            out[col] = pd.to_datetime(out[col], format=fmt, errors="coerce")
    return out

# ---------- 配置式清洗入口 ----------
def apply_cleaning(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    支持：
      - enforce_schema: {dtypes, rename, required}
      - basic_clean: true/false 或 {drop_cols: [...]}
      - fill_na: {num: 'median'|'mean'|..., cat: 'most_frequent'|'constant' }
      - drop_constant: {threshold_unique: 1}
      - clip_outliers: {cols: [...], method: 'iqr'|'z', z_thresh: 3.0, iqr_k: 1.5}
      - bucket_rare: {cols: [...], min_freq: 10, other_label: '_OTHER'}
      - parse_dates: {col: 'auto'|'%Y-%m-%d', ...}
    """
    out = df.copy()
    if not cfg:
        return out

    if cfg.get("enforce_schema"):
        es = cfg["enforce_schema"]
        out = enforce_schema(out,
                             dtypes=es.get("dtypes"),
                             rename=es.get("rename"),
                             required=es.get("required"))
    bc = cfg.get("basic_clean")
    if bc:
        drop_cols = bc.get("drop_cols") if isinstance(bc, dict) else None
        out = basic_clean(out, drop_cols=drop_cols)

    if cfg.get("fill_na"):
        na = cfg["fill_na"]
        out = fill_na(out, num_strategy=na.get("num", "median"),
                           cat_strategy=na.get("cat", "most_frequent"))

    if cfg.get("drop_constant"):
        dc = cfg["drop_constant"]
        out = drop_constant_cols(out, threshold_unique=dc.get("threshold_unique", 1))

    if cfg.get("clip_outliers"):
        co = cfg["clip_outliers"]
        out = clip_outliers(out,
                            cols=co.get("cols"),
                            method=co.get("method", "iqr"),
                            z_thresh=co.get("z_thresh", 3.0),
                            iqr_k=co.get("iqr_k", 1.5))

    if cfg.get("bucket_rare"):
        br = cfg["bucket_rare"]
        out = bucket_rare_categories(out,
                                     cols=br.get("cols", []),
                                     min_freq=br.get("min_freq", 10),
                                     other_label=br.get("other_label", "_OTHER"))

    if cfg.get("parse_dates"):
        out = parse_dates(out, cfg["parse_dates"])

    return out
