from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    mean_squared_error, r2_score,
)

from .model_zoo import get_model
from ..features.transform import build_tabular_preprocess

from pathlib import Path
import inspect

# —— 简单任务自动识别
def _infer_task(y: pd.Series) -> str:
    nunq = y.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(y):
        if pd.api.types.is_float_dtype(y):
            y_nonnull = y.dropna()
            if len(y_nonnull) > 0:
                as_int = (y_nonnull.round() == y_nonnull).mean()
                if as_int > 0.98 and nunq <= max(20, int(len(y)*0.05)):
                    return "classification"
        if nunq <= max(20, int(len(y)*0.01)):
            return "classification"
        return "regression"
    else:
        return "classification"

# —— 默认搜索空间（可选）
_DEFAULT_PARAM_GRID = {
    "lgbm": {
        "model__n_estimators": [100, 200, 400],
        "model__learning_rate": [0.05, 0.1],
        "model__num_leaves": [31, 63, 127],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
    "xgb": {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5, 7],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
    "rf": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__max_features": ["sqrt", "log2", None],
    },
}

def build_pipeline(pre, model):
    return Pipeline([("preprocess", pre), ("model", model)])

def _build_preprocessor(df_with_target: pd.DataFrame, target: str, enable: bool):
    """
    兼容不同版本的 build_tabular_preprocess 签名：
    - (df, target=..., enable=...)
    - (df, target_col=..., enable=...)
    - (df, enable=...) / (df)  —— 此时自动剔除目标列，避免把 label 当作特征
    """
    from ..features.transform import build_tabular_preprocess  # 避免循环导入
    try:
        sig = inspect.signature(build_tabular_preprocess)
        params = sig.parameters
        if "target" in params:
            return build_tabular_preprocess(df_with_target, target=target, enable=enable)
        if "target_col" in params:
            return build_tabular_preprocess(df_with_target, target_col=target, enable=enable)
        # 只有 (df, enable=...) 或 (df)：自动剔除目标列后再构建
        base_df = df_with_target.drop(columns=[target], errors="ignore")
        try:
            return build_tabular_preprocess(base_df, enable=enable)
        except TypeError:
            return build_tabular_preprocess(base_df)
    except Exception:
        # 兜底：再尝试 target_col；若仍不行，则剔除目标列直接调用
        try:
            return build_tabular_preprocess(df_with_target, target_col=target, enable=enable)
        except TypeError:
            base_df = df_with_target.drop(columns=[target], errors="ignore")
            return build_tabular_preprocess(base_df)

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
    search: str = "grid",
    n_iter: int = 20,
    scoring: Optional[str] = None,
    param_grid: Optional[str] = None,
    early_stopping: bool = False,
    val_size: float = 0.15,
    es_rounds: int = 50,
    eval_metric: Optional[str] = None,
) -> Tuple[Pipeline, dict]:

    model_params = model_params or {}
    y = df[target]
    X = df.drop(columns=[target])

    task = task_override or _infer_task(y)
    report: Dict = {"task": task}

    # —— 划分训练/测试
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None if task == "regression" else y
    )

    # —— 预处理（构建时传入包含目标列的 df；内部函数会按需剔除）
    pre = _build_preprocessor(pd.concat([Xtr, ytr], axis=1), target=target, enable=preprocess)

    # —— 构建模型
    model = get_model(model_name, task, **model_params)

    # ============ CV 搜索（不与早停混用） ============
    if isinstance(cv, int) and cv >= 2:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        pipe = build_pipeline(pre, model)
        if param_grid and isinstance(param_grid, str) and param_grid.strip().startswith("@"):
            import json
            pg = json.loads(Path(param_grid[1:]).read_text(encoding="utf-8"))
        elif param_grid and isinstance(param_grid, str):
            import json
            pg = json.loads(param_grid)
        else:
            pg = _DEFAULT_PARAM_GRID.get(model_name, {})
        if search == "random":
            searcher = RandomizedSearchCV(
                pipe, pg, n_iter=n_iter, scoring=scoring, cv=cv, n_jobs=None, refit=True, random_state=random_state
            )
        else:
            searcher = GridSearchCV(
                pipe, pg, scoring=scoring, cv=cv, n_jobs=None, refit=True
            )
        searcher.fit(Xtr, ytr)
        best_pipe = searcher.best_estimator_
        yhat = best_pipe.predict(Xte)

        if task == "classification":
            yprob = None
            try:
                yproba = best_pipe.predict_proba(Xte)
                yprob = yproba[:, 1] if (yproba.ndim == 2 and yproba.shape[1] >= 2) else yproba
            except Exception:
                pass
            acc = float(accuracy_score(yte, yhat))
            f1m = float(f1_score(yte, yhat, average="macro"))
            report.update({"accuracy": acc, "f1_macro": f1m})
            if yprob is not None and len(np.unique(yte)) == 2:
                report["roc_auc"] = float(roc_auc_score(yte, yprob))
            report["classification_report"] = classification_report(yte, yhat, output_dict=True)
            report["_y_true"], report["_y_prob"] = yte.tolist(), (yprob.tolist() if yprob is not None else None)
        else:
            rmse = float(mean_squared_error(yte, yhat, squared=False))
            r2 = float(r2_score(yte, yhat))
            report.update({"rmse": rmse, "r2": r2})

        return best_pipe, {**report, "cv": True, "best_params": getattr(searcher, "best_params_", None),
                           "best_score": getattr(searcher, "best_score_", None)}

    # ============ 非 CV：支持早停 ============
    if early_stopping and model_name in ("lgbm", "xgb"):
        Xtr2, Xval, ytr2, yval = train_test_split(
            Xtr, ytr, test_size=val_size, random_state=random_state, stratify=None if task == "regression" else ytr
        )
        # 只用 X 做预处理（不再拼接 y）
        if preprocess:
            Xtr2_t = pre.fit_transform(Xtr2, ytr2)
            Xval_t = pre.transform(Xval)
            Xte_t  = pre.transform(Xte)
        else:
            Xtr2_t, Xval_t, Xte_t = Xtr2.values, Xval.values, Xte.values

        fit_kwargs = {}
        if model_name == "lgbm":
            import lightgbm as lgb
            fit_kwargs["eval_set"] = [(Xval_t, yval)]
            if eval_metric: fit_kwargs["eval_metric"] = eval_metric
            fit_kwargs["callbacks"] = [lgb.early_stopping(es_rounds, verbose=False)]
        elif model_name == "xgb":
            fit_kwargs["eval_set"] = [(Xval_t, yval)]
            if eval_metric: fit_kwargs["eval_metric"] = eval_metric
            fit_kwargs["early_stopping_rounds"] = es_rounds
            try:
                if not hasattr(model, "get_xgb_params") or ("verbose" not in model.get_xgb_params()):
                    fit_kwargs["verbose"] = False
            except Exception:
                fit_kwargs["verbose"] = False

        clf = model
        clf.fit(Xtr2_t, ytr2, **fit_kwargs)

        yhat = clf.predict(Xte_t)
        yprob = None
        if task == "classification":
            try:
                proba = clf.predict_proba(Xte_t)
                yprob = proba[:, 1] if (proba.ndim == 2 and proba.shape[1] >= 2) else proba
            except Exception:
                pass

        if task == "classification":
            acc = float(accuracy_score(yte, yhat))
            f1m = float(f1_score(yte, yhat, average="macro"))
            report.update({"accuracy": acc, "f1_macro": f1m})
            if yprob is not None and len(np.unique(yte)) == 2:
                report["roc_auc"] = float(roc_auc_score(yte, yprob))
            report["classification_report"] = classification_report(yte, yhat, output_dict=True)
            report["_y_true"], report["_y_prob"] = yte.tolist(), (yprob.tolist() if yprob is not None else None)
        else:
            rmse = float(mean_squared_error(yte, yhat, squared=False))
            r2 = float(r2_score(yte, yhat))
            report.update({"rmse": rmse, "r2": r2})

        if hasattr(clf, "best_iteration_"):
            report["best_iteration"] = int(clf.best_iteration_)
        ev = None
        if hasattr(clf, "evals_result_"):
            ev = clf.evals_result_
        elif hasattr(clf, "evals_result"):
            try: ev = clf.evals_result()
            except Exception: ev = None
        if ev: report["evals_result"] = ev

        pipe = build_pipeline(pre, clf)
        return pipe, report

    # ============ 普通训练（无早停） ============
    pipe = build_pipeline(pre, model)
    pipe.fit(Xtr, ytr)

    yhat = pipe.predict(Xte)
    if task == "classification":
        yprob = None
        try:
            proba = pipe.predict_proba(Xte)
            yprob = proba[:, 1] if (proba.ndim == 2 and proba.shape[1] >= 2) else proba
        except Exception:
            pass
        acc = float(accuracy_score(yte, yhat))
        f1m = float(f1_score(yte, yhat, average="macro"))
        report.update({"accuracy": acc, "f1_macro": f1m})
        if yprob is not None and len(np.unique(yte)) == 2:
            report["roc_auc"] = float(roc_auc_score(yte, yprob))
        report["classification_report"] = classification_report(yte, yhat, output_dict=True)
        report["_y_true"], report["_y_prob"] = yte.tolist(), (yprob.tolist() if yprob is not None else None)
    else:
        rmse = float(mean_squared_error(yte, yhat, squared=False))
        r2 = float(r2_score(yte, yhat))
        report.update({"rmse": rmse, "r2": r2})

    return pipe, report
