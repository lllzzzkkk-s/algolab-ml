from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, precision_recall_curve, auc
)

def infer_task_from_target(y) -> str:
    y_arr = np.asarray(y)
    n_unique = len(np.unique(y_arr))
    # float 近似整数 + 少类别 => 也视为分类
    if np.issubdtype(y_arr.dtype, np.floating):
        if np.allclose(y_arr, np.round(y_arr)) and n_unique <= 20:
            return "classification"
    if n_unique <= 20:
        return "classification"
    return "regression"

def classification_report_dict(y_true, y_pred, y_proba=None) -> Dict:
    out = {
        "task": "classification",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        if y_proba is not None:
            if y_proba.ndim == 1:  # 二分类概率(正类)
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            else:  # 多分类概率矩阵
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        pass
    return out

def regression_report_dict(y_true, y_pred) -> Dict:
    return {
        "task": "regression",
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def pr_curve_auc(y_true, y_score) -> Tuple[np.ndarray, np.ndarray, float]:
    p, r, _ = precision_recall_curve(y_true, y_score)
    return p, r, float(auc(r, p))
