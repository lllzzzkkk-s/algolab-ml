from __future__ import annotations
import inspect
from typing import Dict, List

# sklearn 基础模型
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)

# LightGBM
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# CatBoost
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False


# —— 列出可用模型（按任务）
def available_models(task: str) -> List[str]:
    if task not in ("classification", "regression"):
        raise KeyError(f"unknown task: {task}")
    base_cls = ["logreg", "rf", "gbdt"]
    base_reg = ["ridge", "lasso", "rf", "gbdt"]
    out = (base_cls if task == "classification" else base_reg)
    if _HAS_LGBM: out += ["lgbm"]
    if _HAS_XGB:  out += ["xgb"]
    if _HAS_CAT:  out += ["catboost"]
    return sorted(out)


# —— 别名（任务相关）
def aliases_for_task(task: str) -> Dict[str, str]:
    # 公共别名
    common = {
        "lgb": "lgbm", "lightgbm": "lgbm",
        "xg": "xgb",   "xgboost":  "xgb",
        "cat": "catboost", "cb": "catboost",
        "random_forest": "rf",
        "gb": "gbdt", "gradient_boosting": "gbdt",
    }
    if task == "classification":
        task_only = {"logistic": "logreg", "lr": "logreg"}
    elif task == "regression":
        task_only = {}
    else:
        task_only = {}
    out = dict(common)
    out.update(task_only)
    return out


# —— 归一化模型名：别名 → 规范名
def _normalize_name(name: str, task: str) -> str:
    name = (name or "").lower().strip()
    alias = aliases_for_task(task)
    if name in alias:
        return alias[name]
    return name


# —— 返回构造参数签名（给 --show-params 用）
def model_signature(name: str, task: str):
    cls_map = _class_map(task)
    norm = _normalize_name(name, task)
    if norm not in cls_map:
        raise KeyError(f"Unknown model '{name}' for task '{task}'")
    cls = cls_map[norm]
    sig = inspect.signature(cls.__init__)
    # 去掉 self
    params = [p.name for p in sig.parameters.values() if p.name != "self"]
    return {
        "class": f"{cls.__module__}.{cls.__name__}",
        "init_params": params
    }


# —— 每个任务下的类映射（不含别名）
def _class_map(task: str) -> Dict[str, type]:
    if task == "classification":
        m = {
            "logreg": LogisticRegression,
            "rf": RandomForestClassifier,
            "gbdt": GradientBoostingClassifier,
        }
        if _HAS_LGBM: m["lgbm"] = LGBMClassifier
        if _HAS_XGB:  m["xgb"]  = XGBClassifier
        if _HAS_CAT:  m["catboost"] = CatBoostClassifier
        return m
    elif task == "regression":
        m = {
            "ridge": Ridge,
            "lasso": Lasso,
            "rf": RandomForestRegressor,
            "gbdt": GradientBoostingRegressor,
        }
        if _HAS_LGBM: m["lgbm"] = LGBMRegressor
        if _HAS_XGB:  m["xgb"]  = XGBRegressor
        if _HAS_CAT:  m["catboost"] = CatBoostRegressor
        return m
    else:
        raise KeyError(f"unknown task: {task}")


# —— 构造模型实例
def get_model(name: str, task: str, **params):
    norm = _normalize_name(name, task)
    cls_map = _class_map(task)
    if norm not in cls_map:
        avail = available_models(task)
        raise KeyError(f"Unknown model alias '{name}' for task '{task}'. Available: {avail}")

    # 给不同模型一个合理的默认值（可被 **params 覆盖）
    defaults: Dict[str, object] = {}
    if norm == "logreg":
        defaults = {"max_iter": 1000, "n_jobs": None, "solver": "lbfgs"}
    elif norm in ("ridge", "lasso"):
        defaults = {}
    elif norm == "rf":
        # 分类/回归内核不同，但超参相同；交给 sklearn 处理
        defaults = {"n_estimators": 300, "n_jobs": None, "random_state": 42}
    elif norm == "gbdt":
        defaults = {"random_state": 42}
    elif norm == "lgbm":
        if not _HAS_LGBM:
            raise ImportError("lightgbm 未安装，请 `pip install lightgbm`")
        defaults = {"n_estimators": 300, "learning_rate": 0.05, "n_jobs": None}
    elif norm == "xgb":
        if not _HAS_XGB:
            raise ImportError("xgboost 未安装，请 `pip install xgboost`")
        # 使用较稳妥的默认
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "n_jobs": None,
            "random_state": 42,
            "tree_method": "hist",  # 更快更稳
        }
        # 二分类默认使用概率
        if task == "classification":
            defaults.setdefault("use_label_encoder", False)
            defaults.setdefault("eval_metric", "logloss")
    elif norm == "catboost":
        if not _HAS_CAT:
            raise ImportError("catboost 未安装，请 `pip install catboost`")
        # 关闭多余输出 & 不写入本地目录
        defaults = {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "random_state": 42,
            "verbose": False,
            "allow_writing_files": False,
        }

    cls = cls_map[norm]
    final = {**defaults, **params}
    return cls(**final)
