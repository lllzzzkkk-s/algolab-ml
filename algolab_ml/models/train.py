# algolab_ml/models/train.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    mean_squared_error, r2_score,
)
from sklearn.utils.class_weight import compute_class_weight

from .model_zoo import get_model
from ..features.transform import build_tabular_preprocess

from pathlib import Path
import inspect

# —— 简单任务自动识别
def _infer_task(y: pd.Series) -> str:
    nunq = y.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(y):
        if pd.api.types.is_float_dtype(y):
            as_int = (y.dropna().round() == y.dropna()).mean()
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
    # 注意：最终导出仍然用 sklearn.Pipeline 包裹（哪怕我们在内部用矩阵拟合）
    from sklearn.pipeline import Pipeline
    return Pipeline([("preprocess", pre), ("model", model)])


def _build_preprocessor(df_with_target: pd.DataFrame, target: str, enable: bool):
    """
    兼容不同版本的 build_tabular_preprocess 签名，并且
    ——关键修复：用不含 target 列的 X 来构建预处理器，避免在 CV/Pipeline 中引用 'label'。
    """
    from ..features.transform import build_tabular_preprocess  # 延迟导入避免循环

    # 🚑 关键：剔除 target 列后再交给构建函数
    dfX = df_with_target.drop(columns=[target], errors="ignore")

    try:
        sig = inspect.signature(build_tabular_preprocess)
        params = sig.parameters
        if "target" in params:
            # 传 target 仅用于一些函数内部需要，但真正用于列推断的是 dfX（无 target）
            return build_tabular_preprocess(dfX, target=target, enable=enable)
        if "target_col" in params:
            return build_tabular_preprocess(dfX, target_col=target, enable=enable)
        try:
            return build_tabular_preprocess(dfX, enable=enable)
        except TypeError:
            return build_tabular_preprocess(dfX)
    except Exception:
        try:
            return build_tabular_preprocess(dfX, target_col=target, enable=enable)
        except TypeError:
            return build_tabular_preprocess(dfX)


def _maybe_set_class_weight_param(model, class_weight: str):
    """
    若底层估计器支持 class_weight，则直接设置；否则忽略。
    """
    if not class_weight or class_weight == "none":
        return
    try:
        params = model.get_params(deep=True)
        if "class_weight" in params:
            model.set_params(class_weight=class_weight)
    except Exception:
        pass


def _compute_sample_weights(y: pd.Series,
                            sample_weight_col_values: Optional[pd.Series],
                            class_weight: str) -> Optional[np.ndarray]:
    """
    合成最终 sample_weight:
    - 若 class_weight = balanced / balanced_subsample：按频次计算每类权重（多分类也支持）
    - 与 sample_weight_col（若提供）相乘
    """
    w = None
    if class_weight and class_weight != "none":
        classes = np.unique(y)
        try:
            cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
            cw_map = {cls: weight for cls, weight in zip(classes, cw)}
            w = y.map(cw_map).astype("float64").values
        except Exception:
            w = None

    if sample_weight_col_values is not None:
        sw = pd.to_numeric(sample_weight_col_values, errors="coerce").fillna(0.0).values
        if w is None:
            w = sw
        else:
            w = w * sw

    return w


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
    # 第6步新增
    sample_weight_col: Optional[str] = None,
    class_weight: str = "none",            # none|balanced|balanced_subsample
    smote: bool = False,
) -> Tuple[object, dict]:

    import warnings
    model_params = model_params or {}
    y = df[target]
    X = df.drop(columns=[target])

    task = task_override or _infer_task(y)
    report: Dict = {"task": task}

    # 合成样本权重（先不考虑 SMOTE；SMOTE 时另行处理）
    sample_weight_series = None
    if sample_weight_col and sample_weight_col in df.columns:
        sample_weight_series = df[sample_weight_col]
        report["sample_weight_col"] = sample_weight_col

    # —— 划分训练/测试
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None if task == "regression" else y
    )

    # 训练阶段先拟合预处理器
    pre = _build_preprocessor(pd.concat([Xtr, ytr], axis=1), target=target, enable=preprocess)

    # 构建模型并尽量设置 class_weight（若模型支持）
    model = get_model(model_name, task, **model_params)
    _maybe_set_class_weight_param(model, class_weight)

    # 计算训练集样本权重（在后续分支里切子集/重采样）
    base_weight = _compute_sample_weights(
        ytr,
        sample_weight_series.loc[ytr.index] if sample_weight_series is not None else None,
        class_weight=class_weight,
    )
    if base_weight is not None:
        report["weights_stats"] = {
            "n_nonnull": int(np.isfinite(base_weight).sum()),
            "min": float(np.nanmin(base_weight)),
            "max": float(np.nanmax(base_weight)),
            "mean": float(np.nanmean(base_weight)),
            "sum": float(np.nansum(base_weight)),
        }

    # ============ CV 搜索 ============
    if isinstance(cv, int) and cv >= 2:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("preprocess", pre), ("model", model)])

        # SMOTE 与 CV：当前版本跳过（避免权重与样本数不一致的复杂性）
        if smote:
            warnings.warn("已启用 CV：当前版本在 CV 中不应用 SMOTE（将继续无 SMOTE 进行搜索）。")

        # 搜索空间
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

        fit_kwargs = {}
        if base_weight is not None:
            # Pipeline 下需用 step 前缀
            fit_kwargs["model__sample_weight"] = base_weight

        searcher.fit(Xtr, ytr, **fit_kwargs)
        best_pipe = searcher.best_estimator_
        yhat = best_pipe.predict(Xte)

        if task == "classification":
            # 尝试概率
            yprob = None
            try:
                yproba = best_pipe.predict_proba(Xte)
                if yproba.ndim == 2 and yproba.shape[1] >= 2:
                    yprob = yproba[:, 1]
                else:
                    yprob = yproba
            except Exception:
                pass
            acc = float(accuracy_score(yte, yhat))
            f1m = float(f1_score(yte, yhat, average="macro"))
            report.update({"accuracy": acc, "f1_macro": f1m})
            if yprob is not None and len(np.unique(yte)) == 2:
                report["roc_auc"] = float(roc_auc_score(yte, yprob))
            report["classification_report"] = classification_report(yte, yhat, output_dict=True)
            report["_y_true"], report["_y_pred"] = yte.tolist(), yhat.tolist()
            report["_y_prob"] = (yprob.tolist() if yprob is not None else None)
        else:
            rmse = float(mean_squared_error(yte, yhat, squared=False))
            r2 = float(r2_score(yte, yhat))
            report.update({"rmse": rmse, "r2": r2})

        # 将带 cv_results_ 的对象返回，供 save_cv_results 使用
        return best_pipe, {**report, "cv": True, "best_params": getattr(searcher, "best_params_", None),
                           "best_score": getattr(searcher, "best_score_", None)}

    # ============ 早停分支（lgbm / xgb / catboost） ============
    if early_stopping and model_name in ("lgbm", "xgb", "catboost"):
        from sklearn.exceptions import NotFittedError

        Xtr2, Xval, ytr2, yval = train_test_split(
            Xtr, ytr, test_size=val_size, random_state=random_state, stratify=None if task == "regression" else ytr
        )
        # 拟合预处理器
        Xtr2_t = pre.fit_transform(pd.concat([Xtr2, ytr2], axis=1), ytr2) if preprocess else Xtr2.values
        Xval_t = pre.transform(pd.concat([Xval, yval], axis=1)) if preprocess else Xval.values
        Xte_t  = pre.transform(pd.concat([Xte,  yte],  axis=1)) if preprocess else Xte.values

        # 训练子集的样本权重
        w_tr2 = None
        if base_weight is not None:
            w_tr2 = base_weight[ytr2.index] if isinstance(base_weight, pd.Series) else base_weight
            # 注意：上一步 base_weight 已是 ndarray；这里直接根据 ytr2 的位置切不方便
            # 简化处理：重算一遍更稳妥
            w_tr2 = _compute_sample_weights(
                ytr2,
                sample_weight_series.loc[ytr2.index] if sample_weight_series is not None else None,
                class_weight=class_weight,
            )

        # SMOTE（仅对训练子集）
        if smote and task == "classification":
            try:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=random_state)
                Xtr2_t, ytr2 = sm.fit_resample(Xtr2_t, ytr2)
                if w_tr2 is not None:
                    # 早停路径下无法安全对齐权重与 SMOTE 的合成样本，忽略权重并提示
                    print("⚠️  已启用 SMOTE（早停分支）：训练样本权重将被忽略。")
                w_tr2 = None
            except Exception as e:
                print(f"⚠️  SMOTE 失败（已跳过）：{e}")

        # 适配早停参数
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
            if hasattr(model, "get_xgb_params"):
                if "verbose" not in model.get_xgb_params():
                    fit_kwargs["verbose"] = False
        elif model_name == "catboost":
            # CatBoost: eval_set=(X_val,y_val) & verbose
            fit_kwargs["eval_set"] = (Xval_t, yval)
            if eval_metric:
                fit_kwargs["eval_metric"] = eval_metric
            fit_kwargs["verbose"] = False

        # 拟合
        clf = model
        try:
            clf.fit(Xtr2_t, ytr2, sample_weight=w_tr2, **fit_kwargs)
        except TypeError:
            # 某些模型不接受 sample_weight
            clf.fit(Xtr2_t, ytr2, **fit_kwargs)

        # 推理
        yhat = clf.predict(Xte_t)
        yprob = None
        if task == "classification":
            try:
                proba = clf.predict_proba(Xte_t)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    yprob = proba[:, 1]
                else:
                    yprob = proba
            except Exception:
                pass

        # 评估
        if task == "classification":
            acc = float(accuracy_score(yte, yhat))
            f1m = float(f1_score(yte, yhat, average="macro"))
            report.update({"accuracy": acc, "f1_macro": f1m})
            if yprob is not None and len(np.unique(yte)) == 2:
                report["roc_auc"] = float(roc_auc_score(yte, yprob))
            report["classification_report"] = classification_report(yte, yhat, output_dict=True)
            report["_y_true"], report["_y_pred"] = yte.tolist(), yhat.tolist()
            report["_y_prob"] = (yprob.tolist() if yprob is not None else None)
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
            try:
                ev = clf.evals_result()
            except Exception:
                ev = None
        if ev:
            report["evals_result"] = ev

        pipe = build_pipeline(pre, clf)
        return pipe, report

    # ============ 普通训练（无早停/无 CV） ============
    # 统一走“预处理→（可选 SMOTE）→ 拟合底模”的路径，确保权重/SMOTE 都稳定可控
    Xtr_t = pre.fit_transform(pd.concat([Xtr, ytr], axis=1), ytr) if preprocess else Xtr.values
    Xte_t = pre.transform(pd.concat([Xte, yte], axis=1)) if preprocess else Xte.values

    w_tr = base_weight
    if smote and task == "classification":
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=random_state)
            Xtr_t, ytr = sm.fit_resample(Xtr_t, ytr)
            if w_tr is not None:
                print("⚠️  已启用 SMOTE：训练样本权重将被忽略（长度无法与合成样本对齐）。")
            w_tr = None
        except Exception as e:
            print(f"⚠️  SMOTE 失败（已跳过）：{e}")

    clf = model
    try:
        clf.fit(Xtr_t, ytr, sample_weight=w_tr)
    except TypeError:
        clf.fit(Xtr_t, ytr)

    # 推理
    yhat = clf.predict(Xte_t)

    if task == "classification":
        yprob = None
        try:
            proba = clf.predict_proba(Xte_t)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                yprob = proba[:, 1]
            else:
                yprob = proba
        except Exception:
            pass
        acc = float(accuracy_score(yte, yhat))
        f1m = float(f1_score(yte, yhat, average="macro"))
        report.update({"accuracy": acc, "f1_macro": f1m})
        if yprob is not None and len(np.unique(yte)) == 2:
            report["roc_auc"] = float(roc_auc_score(yte, yprob))
        report["classification_report"] = classification_report(yte, yhat, output_dict=True)
        report["_y_true"], report["_y_pred"] = yte.tolist(), yhat.tolist()
        report["_y_prob"] = (yprob.tolist() if yprob is not None else None)
    else:
        rmse = float(mean_squared_error(yte, yhat, squared=False))
        r2 = float(r2_score(yte, yhat))
        report.update({"rmse": rmse, "r2": r2})

    pipe = build_pipeline(pre, clf)
    return pipe, report
