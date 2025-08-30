from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    mean_squared_error, r2_score
)

def infer_task_from_target(y: pd.Series) -> str:
    """
    规则（通俗版）：
    - 数值且唯一值很多 -> 回归，但若“几乎都是整数且种类很少(≤10)”则更像分类
    - 其他 -> 分类
    """
    y = pd.Series(y)
    if pd.api.types.is_numeric_dtype(y):
        nun = y.nunique(dropna=True)
        if nun > max(20, 0.05 * len(y)):
            y_round = np.round(y)
            if np.allclose(y, y_round, atol=1e-6) and len(np.unique(y_round)) <= 10:
                return "classification"
            return "regression"
    return "classification"

def _to_py(obj):
    # 把 numpy 类型递归转为内置 python 类型，便于 JSON 序列化
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _to_py(v) for v in obj ]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj

def classification_report_dict(y_true, y_pred, y_prob=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "task": "classification",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": _to_py(classification_report(y_true, y_pred, output_dict=True)),
    }
    try:
        labels = np.unique(y_true)
        if len(labels) == 2 and y_prob is not None:
            # 概率矩阵 -> 取正类概率列
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                prob_pos = y_prob[:, 1]
            else:
                prob_pos = y_prob
            out["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
    except Exception:
        pass
    return out

def regression_report_dict(y_true, y_pred) -> Dict[str, Any]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"task": "regression", "rmse": rmse, "r2": r2}
