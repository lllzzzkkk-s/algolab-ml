# algolab_ml/pipelines/tabular.py
from __future__ import annotations
import pandas as pd

from algolab_ml.models.train import fit_tabular


def run_df(
    df: pd.DataFrame,
    *,
    target: str,
    model: str,
    test_size: float = 0.2,
    random_state: int = 42,
    preprocess: bool = True,
    model_params: dict | None = None,
    task: str | None = None,
    cv: int = 0,
    search: str = "grid",
    n_iter: int = 20,
    scoring: str | None = None,
    param_grid: str | None = None,
    early_stopping: bool = False,
    val_size: float = 0.15,
    es_rounds: int = 50,
    eval_metric: str | None = None,
    # 第6步新增
    sample_weight_col: str | None = None,
    class_weight: str = "none",            # none|balanced|balanced_subsample
    smote: bool = False,
):
    """
    简单薄封装，直接转给 fit_tabular。
    """
    pipe, report = fit_tabular(
        df=df,
        target=target,
        model_name=model,
        test_size=test_size,
        random_state=random_state,
        preprocess=preprocess,
        model_params=model_params or {},
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
        # 新增
        sample_weight_col=sample_weight_col,
        class_weight=class_weight,
        smote=smote,
    )
    return pipe, report
