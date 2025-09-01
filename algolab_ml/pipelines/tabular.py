from __future__ import annotations
import pandas as pd
from typing import Tuple, Optional, Dict
from ..models.train import fit_tabular


def run_df(
    df: pd.DataFrame,
    target: str,
    model: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    preprocess: bool = True,
    model_params: Optional[Dict] = None,
    task: Optional[str] = None,
    cv: int = 0,
    search: str = "grid",
    n_iter: int = 20,
    scoring: Optional[str] = None,
    param_grid: Optional[str] = None,
    early_stopping: bool = False,
    val_size: float = 0.15,
    es_rounds: int = 50,
    eval_metric: Optional[str] = None,
    # 第 6 步：样本/不平衡
    sample_weight: Optional[pd.Series] = None,
    class_weight: Optional[str] = None,   # "balanced" / None
    # 第 7 步：阈值调优
    optimize_metric: Optional[str] = None,
    threshold_grid: Optional[str] = "auto",
) -> Tuple[object, dict]:

    pipe, report = fit_tabular(
        df=df,
        target=target,
        model_name=model,
        test_size=test_size,
        random_state=random_state,
        preprocess=preprocess,
        model_params=(model_params or {}),
        task_override=task,
        cv=cv,
        search=search,
        n_iter=n_iter,
        scoring=scoring,
        param_grid=param_grid,
        early_stopping=early_stopping,
        val_size=val_size,
        es_rounds=es_rounds,
        eval_metric=eval_metric,
        sample_weight=(sample_weight.values if isinstance(sample_weight, pd.Series) else sample_weight),
        class_weight=class_weight,
        optimize_metric=optimize_metric,
        threshold_grid=threshold_grid,
    )
    return pipe, report
