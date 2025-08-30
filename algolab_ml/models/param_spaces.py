from __future__ import annotations
from typing import Dict, Any

def default_param_grid(model_key: str, task: str) -> Dict[str, Any]:
    k = model_key.lower()
    if k == "lgbm":
        return {
            "model__n_estimators": [100, 300, 600],
            "model__learning_rate": [0.1, 0.05, 0.02],
            "model__num_leaves": [31, 63, 127],
            "model__subsample": [1.0, 0.9, 0.8],
            "model__colsample_bytree": [1.0, 0.9, 0.8],
        }
    if k == "xgb":
        return {
            "model__n_estimators": [200, 500],
            "model__max_depth": [3, 6, 8],
            "model__learning_rate": [0.1, 0.05],
            "model__subsample": [1.0, 0.8],
            "model__colsample_bytree": [1.0, 0.8],
        }
    if k == "rf":
        return {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10, 20],
        }
    if k in ("logreg", "ridge", "lasso"):
        return {"model__C" if k == "logreg" else "model__alpha": [0.1, 1.0, 10.0]}
    return {}
