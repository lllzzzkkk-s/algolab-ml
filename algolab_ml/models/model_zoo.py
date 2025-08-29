from __future__ import annotations
import inspect
from typing import Dict, Type

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# ---- 常见别名 → 规范名 ----
_CANON: Dict[str, str] = {
    # boosting
    "xgboost": "xgb", "xgb": "xgb",
    "lightgbm": "lgbm", "lgb": "lgbm", "lgbm": "lgbm",
    "cat": "catboost", "cb": "catboost", "catboost": "catboost",
    # forest
    "randomforest": "rf", "rf": "rf",
    # gbdt
    "gbm": "gbdt", "gboost": "gbdt", "gradientboosting": "gbdt", "gbdt": "gbdt",
    # linear / logistic
    "logistic": "logreg", "logreg": "logreg",
    "ridge": "ridge", "lasso": "lasso",
}

# ---- 注册表 ----
_CLASSIFIERS: Dict[str, Type] = {
    "logreg": LogisticRegression,
    "rf": RandomForestClassifier,
    "gbdt": GradientBoostingClassifier,
    "xgb": XGBClassifier,
    "lgbm": LGBMClassifier,
    "catboost": CatBoostClassifier,
}
_REGRESSORS: Dict[str, Type] = {
    "ridge": Ridge,
    "lasso": Lasso,
    "rf": RandomForestRegressor,
    "gbr": GradientBoostingRegressor,
    "xgb": XGBRegressor,
    "lgbm": LGBMRegressor,
    "catboost": CatBoostRegressor,
}

def _canon(name: str) -> str:
    return _CANON.get(name.lower(), name.lower())

def registry(task: str) -> Dict[str, Type]:
    return _CLASSIFIERS if task == "classification" else _REGRESSORS

def available_models(task: str) -> list[str]:
    return sorted(registry(task).keys())

def aliases_for_task(task: str) -> Dict[str, str]:
    """仅返回对应 task 可用的别名映射（alias -> canonical）。"""
    valid = set(available_models(task))
    return {a: c for a, c in _CANON.items() if c in valid}

def model_class(name: str, task: str):
    name = _canon(name)
    table = registry(task)
    if name not in table:
        raise KeyError(f"Unknown model alias '{name}'. Try one of: {sorted(table.keys())} "
                       f"(aliases accepted: {sorted(set(aliases_for_task(task).keys()))})")
    return table[name]

def model_signature(name: str, task: str) -> str:
    cls = model_class(name, task)
    sig = inspect.signature(cls.__init__)
    return f"{cls.__name__}{sig}"

def get_model(name: str, task: str = "classification", **kwargs):
    cls = model_class(name, task)
    # CatBoost 默认安静
    if cls in (CatBoostClassifier, CatBoostRegressor) and "verbose" not in kwargs:
        kwargs["verbose"] = False
    return cls(**kwargs)
