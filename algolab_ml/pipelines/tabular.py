from __future__ import annotations
import pandas as pd
from algolab_ml.models.train import fit_tabular

def run(csv_path: str, target: str, model: str="xgb", test_size: float=0.2,
        random_state: int=42, preprocess: bool=True, **model_params):
    df = pd.read_csv(csv_path)
    return fit_tabular(df, target=target, model_name=model, test_size=test_size,
                       random_state=random_state, preprocess=preprocess,
                       model_params=model_params or None)

def run_df(df: pd.DataFrame, target: str, model: str="xgb", test_size: float=0.2,
           random_state: int=42, preprocess: bool=True, **model_params):
    return fit_tabular(df, target=target, model_name=model, test_size=test_size,
                       random_state=random_state, preprocess=preprocess,
                       model_params=model_params or None)
