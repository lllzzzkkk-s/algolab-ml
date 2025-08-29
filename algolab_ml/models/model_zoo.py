# algolab_ml/models/model_zoo.py
from __future__ import annotations
import inspect
from typing import Dict, Type

# 这些是 sklearn 自带的，直接安全导入
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

# 不要在模块顶层导入 xgboost / lightgbm / catboost（避免强依赖）
# 改为在实际使用时再导入，并给出缺库的友好提示

# ---- 常见别名 → 规范名 ----
_CANON: Dict[str, str] = {
    # boosting
    "xgboost": "xgb", "xgb": "xgb",
    "lightgbm": "lgbm", "lgb": "lgbm", "lgbm": "lgbm",
    "cat": "catboost", "cb": "catboost", "catboost": "catboost",
    # forest
    "randomforest": "rf", "rf": "rf",
    # gbdt（分类用 gbdt，回归用 gbr）
    "gbm": "gbdt", "gboost": "gbdt", "gradientboosting": "gbdt", "gbdt": "gbdt",
    # linear / logistic
    "logistic": "logreg", "logreg": "logreg",
    "ridge": "ridge", "lasso": "lasso",
}

# ---- sklearn 部分直接映射 ----
_SK_CLASSIFIERS: Dict[str, Type] = {
    "logreg": LogisticRegression,
    "rf": RandomForestClassifier,
    "gbdt": GradientBoostingClassifier,
}
_SK_REGRESSORS: Dict[str, Type] = {
    "ridge": Ridge,
    "lasso": Lasso,
    "rf": RandomForestRegressor,
    "gbr": GradientBoostingRegressor,
}

# 其余三类为可选依赖：xgb / lgbm / catboost
_OPTIONAL_MODELS = {"xgb", "lgbm", "catboost"}


def _canon(name: str) -> str:
    return _CANON.get(name.lower(), name.lower())


def available_models(task: str) -> list[str]:
    task = task.lower()
    if task == "classification":
        base = set(_SK_CLASSIFIERS.keys())
    else:
        base = set(_SK_REGRESSORS.keys())
    # 列表中包含可选模型名称（即便未安装也显示出来）
    return sorted(base | _OPTIONAL_MODELS)


def aliases_for_task(task: str) -> Dict[str, str]:
    """仅返回对应 task 可用的别名映射（alias -> canonical）。"""
    valid = set(available_models(task))
    return {a: c for a, c in _CANON.items() if c in valid}


# ---- 可选依赖的按需导入 ----
def _import_xgb(task: str):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception as e:
        raise RuntimeError(
            "需要使用 XGBoost，请先安装：\n"
            "  pip install xgboost\n"
            "或 conda：\n"
            "  conda install -c conda-forge xgboost"
        ) from e
    return XGBClassifier if task == "classification" else XGBRegressor


def _import_lgbm(task: str):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except Exception as e:
        raise RuntimeError(
            "需要使用 LightGBM，请先安装：\n"
            "  pip install lightgbm\n"
            "或 conda：\n"
            "  conda install -c conda-forge lightgbm"
        ) from e
    return LGBMClassifier if task == "classification" else LGBMRegressor


def _import_catboost(task: str):
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except Exception as e:
        raise RuntimeError(
            "需要使用 CatBoost，请先安装：\n"
            "  pip install 'catboost==1.2.8'\n"
            "或 conda：\n"
            "  conda install -c conda-forge catboost=1.2.8"
        ) from e
    return CatBoostClassifier if task == "classification" else CatBoostRegressor


def model_class(name: str, task: str):
    """
    返回对应模型类；对可选依赖（xgb/lgbm/catboost）在此处按需导入。
    """
    task = task.lower()
    cname = _canon(name)
    valid = set(available_models(task))
    if cname not in valid:
        raise KeyError(
            f"Unknown model alias '{name}'. "
            f"Try one of: {sorted(valid)} "
            f"(aliases accepted: {sorted(set(aliases_for_task(task).keys()))})"
        )

    if task == "classification":
        if cname in _SK_CLASSIFIERS:
            return _SK_CLASSIFIERS[cname]
    else:
        if cname in _SK_REGRESSORS:
            return _SK_REGRESSORS[cname]

    # 可选依赖按需导入
    if cname == "xgb":
        return _import_xgb(task)
    if cname == "lgbm":
        return _import_lgbm(task)
    if cname == "catboost":
        return _import_catboost(task)

    # 理论到不了这里
    raise KeyError(f"Model '{cname}' not implemented for task '{task}'")


def model_signature(name: str, task: str) -> str:
    cls = model_class(name, task)
    sig = inspect.signature(cls.__init__)
    return f"{cls.__name__}{sig}"


def get_model(name: str, task: str = "classification", **kwargs):
    cname = _canon(name)
    cls = model_class(cname, task)
    # CatBoost 默认更安静一些
    if cname == "catboost" and "verbose" not in kwargs:
        kwargs["verbose"] = False
    return cls(**kwargs)
