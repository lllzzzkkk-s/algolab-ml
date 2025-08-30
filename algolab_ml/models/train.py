from __future__ import annotations
from typing import Dict, Tuple, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from .model_zoo import get_model
from ..utils.metrics import (
    infer_task_from_target, classification_report_dict, regression_report_dict
)
from .param_spaces import default_param_grid

def build_tabular_preprocess(df: pd.DataFrame, target: str, enable: bool=True):
    if not enable:
        return "passthrough", []
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_tf = Pipeline([("scaler", StandardScaler(with_mean=False))])
    cat_tf = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    pre = ColumnTransformer(
        transformers=[("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre, num_cols + cat_cols

def fit_tabular(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    test_size: float=0.2,
    random_state: int=42,
    preprocess: bool=True,
    task_override: Optional[str]=None,
    model_params: Optional[Dict]=None,
    cv: Optional[int]=None,
    scoring: Optional[str]=None,
    search: Optional[str]=None,            # "grid" | "random" | None
    param_grid: Optional[Dict]=None,
    n_iter: int=20
) -> Tuple[Pipeline, Dict]:
    # 1) 任务识别（自动/强制）
    y = df[target]
    task = task_override or infer_task_from_target(y)

    # 2) 预处理 + 模型
    pre, _ = build_tabular_preprocess(df, target, enable=preprocess)
    model = get_model(model_name, task=task, **(model_params or {}))
    pipe = Pipeline([("prep", pre), ("model", model)])

    # 3) 切分
    X = df.drop(columns=[target])
    strat = y if task == "classification" else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    # 4) CV / 搜索（只把参数网格给 model__*）
    estimator = pipe
    if cv and (search in ("grid", "random") or param_grid):
        grid = param_grid or default_param_grid(model_name, task)
        if grid:
            if search == "random":
                estimator = RandomizedSearchCV(
                    pipe, grid, n_iter=n_iter, cv=cv, scoring=scoring,
                    n_jobs=1, verbose=0, random_state=random_state
                )
            else:
                estimator = GridSearchCV(
                    pipe, grid, cv=cv, scoring=scoring, n_jobs=1, verbose=0
                )

    # 5) 训练（这里 **不要** 传入 model_params 以外的任何 dict）
    estimator.fit(Xtr, ytr)

    # 6) 评估
    if task == "classification":
        y_pred = estimator.predict(Xte)
        y_proba = None
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(Xte)
            y_proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
        report = classification_report_dict(yte, y_pred, y_proba)
    else:
        y_pred = estimator.predict(Xte)
        report = regression_report_dict(yte, y_pred)

    # 7) 若做了搜索，附带最优参数
    if hasattr(estimator, "best_params_"):
        report["search"] = {
            "best_params": estimator.best_params_,
            "scoring": scoring,
            "cv": cv,
            "search": search or "grid",
        }

    return estimator, report
