from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_tabular_preprocess(X: pd.DataFrame, scale: bool=True, one_hot: bool=True):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if scale and num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if one_hot and cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if not transformers:
        return None
    return ColumnTransformer(transformers=transformers, remainder="drop")
