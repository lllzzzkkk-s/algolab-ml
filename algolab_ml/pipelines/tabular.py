from __future__ import annotations
from typing import Dict, Tuple, Optional
import json
import pandas as pd
from ..models.train import fit_tabular

def run_df(
    df: pd.DataFrame,
    target: str,
    model: str,
    test_size: float = 0.2,
    random_state: int = 42,
    preprocess: bool = True,
    model_params: Dict | None = None,
    task: Optional[str] = None,
    cv: int = 0,
    search: Optional[str] = None,
    n_iter: int = 20,
    scoring: Optional[str] = None,
    param_grid: Optional[str | Dict] = None,
) -> Tuple[object, dict]:
    return fit_tabular(
        df,
        target=target,
        model_name=model,
        test_size=test_size,
        random_state=random_state,
        preprocess=preprocess,
        model_params=model_params or {},
        task_override=task,
        cv=cv, search=search, n_iter=n_iter,
        scoring=scoring, param_grid=param_grid,
    )

def run(
    csv_path: str,
    target: str,
    model: str,
    test_size: float = 0.2,
    random_state: int = 42,
    preprocess: bool = True,
    model_params: Dict | None = None,
    task: Optional[str] = None,
    cv: int = 0,
    search: Optional[str] = None,
    n_iter: int = 20,
    scoring: Optional[str] = None,
    param_grid: Optional[str | Dict] = None,
) -> Tuple[object, dict]:
    df = pd.read_csv(csv_path)
    return run_df(
        df,
        target=target,
        model=model,
        test_size=test_size,
        random_state=random_state,
        preprocess=preprocess,
        model_params=model_params,
        task=task,
        cv=cv, search=search, n_iter=n_iter,
        scoring=scoring, param_grid=param_grid,
    )
