from __future__ import annotations
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error

def classification_metrics(y_true, y_pred, y_proba=None):
    rpt = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_proba is not None:
        try:
            rpt["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
    return rpt

def regression_metrics(y_true, y_pred):
    return {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
