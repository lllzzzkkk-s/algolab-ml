# algolab_ml/models/train.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, classification_report
)

from .model_zoo import get_model
from ..features.transform import build_tabular_preprocess


def train_test_split_df(df: pd.DataFrame, target: str, test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    # 如果很可能是分类，就分层切分
    stratify = y if _is_likely_classification(y) else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def _is_likely_classification(y: pd.Series, max_classes: int = 20) -> bool:
    """用于 stratify 的粗判：‘看起来像分类’就返回 True。"""
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return True
    u = pd.unique(y.dropna())
    if len(u) == 2:
        return True
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= max_classes:
        return True
    if pd.api.types.is_float_dtype(y):
        ints_like = all(float(v).is_integer() for v in u if pd.notna(v))
        if ints_like and y.nunique() <= max_classes:
            return True
    return False


def _task_from_target(y: pd.Series) -> str:
    """自动判定任务类型：二值/少类别/离散 → classification；其余 → regression。"""
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"
    u = pd.unique(y.dropna())
    if len(u) == 2:
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 20:
        return "classification"
    if pd.api.types.is_float_dtype(y):
        ints_like = all(float(v).is_integer() for v in u if pd.notna(v))
        if ints_like and y.nunique() <= 20:
            return "classification"
    return "regression"


def fit_tabular(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    preprocess: bool = True,
    model_params: Dict | None = None,
    task_override: Optional[str] = None,
) -> Tuple[Pipeline, Dict]:
    """训练入口：自动/指定任务 + 评估 + 返回 (pipeline, report)。"""
    Xtr, Xte, ytr, yte = train_test_split_df(df, target, test_size=test_size, random_state=random_state)

    # 任务选择：优先用覆盖，其次自动判定
    task = task_override if task_override in ("classification", "regression") else _task_from_target(ytr)

    # 只在构造模型时解包参数
    params = model_params or {}
    model = get_model(model_name, task=task, **params)

    # 组装 pipeline
    if preprocess:
        pre = build_tabular_preprocess(Xtr)
        pipe = Pipeline([("pre", pre), ("model", model)])
    else:
        pipe = Pipeline([("model", model)])

    # 训练（不要把 model_params 再传给 fit）
    pipe.fit(Xtr, ytr)

    # 评估
    report: Dict = {"task": task}
    if task == "classification":
        ypred = pipe.predict(Xte)
        report["accuracy"] = float(accuracy_score(yte, ypred))
        report["f1_macro"] = float(f1_score(yte, ypred, average="macro"))
        # AUC（支持二分类 / 多分类OvR）
        proba = getattr(pipe, "predict_proba", None)
        if callable(proba):
            p = pipe.predict_proba(Xte)
            if p.ndim == 2 and p.shape[1] == 2:
                report["roc_auc"] = float(roc_auc_score(yte, p[:, 1]))
            else:
                try:
                    report["roc_auc_ovr_macro"] = float(
                        roc_auc_score(yte, p, multi_class="ovr", average="macro")
                    )
                except Exception:
                    pass
        # 可读报告
        try:
            report["classification_report"] = classification_report(yte, ypred, output_dict=True)
        except Exception:
            pass
    else:
        ypred = pipe.predict(Xte)
        report["rmse"] = float(mean_squared_error(yte, ypred, squared=False))
        report["r2"] = float(r2_score(yte, ypred))

    return pipe, report
