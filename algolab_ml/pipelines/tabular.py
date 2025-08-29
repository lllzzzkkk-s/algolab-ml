# algolab_ml/pipelines/tabular.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
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
    )
