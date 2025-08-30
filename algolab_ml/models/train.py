from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import json
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    mean_squared_error, r2_score
)

from .model_zoo import get_model

# ========== 兼容：metrics & param_spaces 可用则用，否则这里内置 ==========
try:
    from ..utils.metrics import infer_task_from_target, classification_report_dict, regression_report_dict
except Exception:
    def infer_task_from_target(y: pd.Series) -> str:
        y = pd.Series(y)
        if pd.api.types.is_numeric_dtype(y):
            # 连续型：唯一值多且非近似整数
            nun = y.nunique(dropna=True)
            if nun > max(20, 0.05 * len(y)):
                # 判断是不是“近似整数且类别很少”的情况（这类更像分类）
                y_round = np.round(y)
                if np.allclose(y, y_round, atol=1e-6) and len(np.unique(y_round)) <= 10:
                    return "classification"
                return "regression"
        return "classification"

    def classification_report_dict(y_true, y_pred, y_prob=None):
        out = {
            "task": "classification",
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "classification_report": json.loads(
                classification_report(y_true, y_pred, output_dict=True).__repr__().replace("'", '"')
            ) if False else classification_report(y_true, y_pred, output_dict=True)
        }
        # 二分类且有概率
        try:
            labels = np.unique(y_true)
            if len(labels) == 2 and y_prob is not None:
                # 取正类概率
                if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                    pos_idx = 1
                    out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, pos_idx]))
        except Exception:
            pass
        return out

    def regression_report_dict(y_true, y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return {"task": "regression", "rmse": rmse, "r2": r2}

try:
    from .param_spaces import default_param_grid as _default_param_grid
except Exception:
    def _default_param_grid(model_name: str, task: str) -> Dict[str, List]:
        # 轻量默认搜索空间（可根据需要扩展）
        if model_name == "lgbm":
            if task == "classification":
                return {"n_estimators": [200, 400], "num_leaves": [31, 63], "learning_rate": [0.1, 0.05]}
            else:
                return {"n_estimators": [200, 400], "num_leaves": [31, 63], "learning_rate": [0.1, 0.05]}
        if model_name == "xgb":
            return {"n_estimators": [200, 400], "max_depth": [4, 6], "learning_rate": [0.1, 0.05]}
        if model_name in ("rf",):
            return {"n_estimators": [200, 500], "max_depth": [None, 10, 20]}
        if model_name in ("gbdt", "gbr"):
            return {"n_estimators": [200, 400], "learning_rate": [0.1, 0.05], "max_depth": [2, 3]}
        if model_name in ("logreg", "ridge", "lasso"):
            return {"C": [0.1, 1.0, 10.0]} if model_name == "logreg" else {"alpha": [0.1, 1.0, 10.0]}
        return {}

# ========== 预处理 ==========
def _make_ohe():
    """OneHotEncoder 兼容封装：优先用 sparse_output（sklearn>=1.4），否则退回 sparse。"""
    try:
        # 新版参数名
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # 旧版参数名
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_tabular_preprocess(df: pd.DataFrame, target: str, enable: bool = True):
    if not enable:
        return None
    # 数值/类别列自动识别
    feature_cols = [c for c in df.columns if c != target]
    num_cols = df[feature_cols].select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    num_pipe = Pipeline([("scaler", StandardScaler())])
    ohe = _make_ohe()

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", ohe,      cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # 强制 densify（遇到稀疏也转为 dense）
    )
    return pre

# ========== 训练 ==========
def fit_tabular(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    preprocess: bool = True,
    model_params: Dict | None = None,
    task_override: Optional[str] = None,
    cv: int = 0,
    search: Optional[str] = None,          # 'grid' | 'random'
    param_grid: Optional[str | Dict] = None,  # JSON str / @file / dict
    n_iter: int = 20,
    scoring: Optional[str] = None,
) -> Tuple[Pipeline, Dict]:
    if target not in df.columns:
        raise KeyError(f"目标列 '{target}' 不存在，当前列：{list(df.columns)[:20]} ...")

    # 1) 任务判断
    y = df[target]
    task = task_override or infer_task_from_target(y)

    # 2) 划分
    X = df.drop(columns=[target])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=(y if task=="classification" else None))

    # 3) 预处理 + 模型
    pre = build_tabular_preprocess(pd.concat([Xtr, ytr], axis=1), target=target, enable=preprocess)
    base_est = get_model(model_name, task=task, **(model_params or {}))
    steps = []
    if pre is not None:
        steps.append(("preprocess", pre))
    steps.append(("model", base_est))
    pipe = Pipeline(steps)

    # 4) 解析 param_grid
    if isinstance(param_grid, str):
        pg = param_grid.strip()
        if pg.startswith("@"):
            pg = json.loads(Path(pg[1:]).read_text(encoding="utf-8"))
        else:
            pg = json.loads(pg)
    elif isinstance(param_grid, dict):
        pg = param_grid
    else:
        pg = _default_param_grid(model_name, task)

    # 将参数名加上 'model__' 前缀（用于 Pipeline）
    if pg:
        pg = {f"model__{k}": v for k, v in pg.items()}

    # 5) 拟合（可选 CV）
    estimator = pipe
    cv_used = None
    if isinstance(cv, int) and cv > 1 and pg:
        cv_used = int(cv)
        if (search or "grid") == "random":
            estimator = RandomizedSearchCV(
                pipe, param_distributions=pg, n_iter=int(n_iter),
                cv=cv_used, scoring=scoring, n_jobs=1, refit=True, random_state=random_state, verbose=0
            )
        else:
            estimator = GridSearchCV(
                pipe, param_grid=pg,
                cv=cv_used, scoring=scoring, n_jobs=1, refit=True, verbose=0
            )

    estimator.fit(Xtr, ytr)
    best_est = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator

    # 6) 预测 & 评估
    if task == "classification":
        y_pred = best_est.predict(Xte)
        y_prob = None
        try:
            proba = best_est.predict_proba(Xte)
            if proba.ndim == 2:
                y_prob = proba
        except Exception:
            pass
        report = classification_report_dict(yte, y_pred, y_prob=y_prob)
        # 给导出阶段画图用的隐藏字段（只保留简短切片）
        try:
            report["_y_true"] = np.asarray(yte).tolist()
            if y_prob is not None:
                report["_y_prob"] = np.asarray(y_prob).tolist()
        except Exception:
            pass
    else:
        y_pred = best_est.predict(Xte)
        report = regression_report_dict(yte, y_pred)

    # 7) 搜索信息
    if hasattr(estimator, "best_params_"):
        report["search"] = {
            "best_params": estimator.best_params_,
            "scoring": scoring,
            "cv": cv_used,
            "search": search or "grid",
        }
        # 结果表
        try:
            report["_cv_results"] = estimator.cv_results_
        except Exception:
            pass

    return estimator, report
