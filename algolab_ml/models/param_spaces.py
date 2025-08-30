from __future__ import annotations
from typing import Dict, List

def default_param_grid(model_name: str, task: str) -> Dict[str, List]:
    """
    轻量级默认搜索空间（可按需扩展/覆盖）。
    用于 --cv --search 时未显式提供 --param-grid 的场景。
    """
    name = model_name.lower()
    t = task.lower()

    if name == "lgbm":
        if t == "classification":
            return {
                "n_estimators": [200, 400, 800],
                "num_leaves": [31, 63, 127],
                "learning_rate": [0.1, 0.05, 0.02],
                "subsample": [1.0, 0.9, 0.8],
                "colsample_bytree": [1.0, 0.9, 0.8],
                "reg_alpha": [0.0, 1.0],
                "reg_lambda": [0.0, 2.0],
            }
        else:  # regression
            return {
                "n_estimators": [300, 600, 1000],
                "num_leaves": [31, 63, 127],
                "learning_rate": [0.1, 0.05, 0.02],
                "subsample": [1.0, 0.9, 0.8],
                "colsample_bytree": [1.0, 0.9, 0.8],
                "reg_alpha": [0.0, 1.0],
                "reg_lambda": [0.0, 2.0],
            }

    if name == "xgb":
        return {
            "n_estimators": [200, 400, 800],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.1, 0.05, 0.02],
            "subsample": [1.0, 0.9, 0.8],
            "colsample_bytree": [1.0, 0.9, 0.8],
            "reg_alpha": [0.0, 1.0],
            "reg_lambda": [0.0, 2.0],
        }

    if name == "rf":
        return {"n_estimators": [200, 500, 800], "max_depth": [None, 10, 20]}

    if name in ("gbdt", "gbr"):
        return {"n_estimators": [200, 400, 800], "learning_rate": [0.1, 0.05, 0.02], "max_depth": [2, 3]}

    if name == "logreg":
        return {"C": [0.1, 1.0, 10.0], "penalty": ["l2"]}

    if name in ("ridge", "lasso"):
        return {"alpha": [0.1, 1.0, 10.0]}

    # 未覆盖模型 -> 空字典（不做搜索）
    return {}
