from __future__ import annotations
from typing import Dict, Tuple, Optional, List
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
    return Pipeline([("preprocess", pre), ("model", model)])


def _build_preprocessor(X: pd.DataFrame, target: str, enable: bool):
    from ..features.transform import build_tabular_preprocess  # 延迟导入
    import inspect
    try:
        sig = inspect.signature(build_tabular_preprocess)
        params = sig.parameters
        if "target" in params:
            # 传入 X（不含 target），把 target 名单独传递，便于对方函数显式排除
            return build_tabular_preprocess(X, target=target, enable=enable)
        if "target_col" in params:
            return build_tabular_preprocess(X, target_col=target, enable=enable)
        try:
            return build_tabular_preprocess(X, enable=enable)
        except TypeError:
            return build_tabular_preprocess(X)
    except Exception:
        try:
            return build_tabular_preprocess(X, target_col=target, enable=enable)
        except TypeError:
            return build_tabular_preprocess(X)


# ========= 阈值调优（第 7 步） =========
def _parse_threshold_grid(grid_spec: Optional[str]) -> np.ndarray:
    """
    grid 语法：
      - None / "auto"：使用 np.linspace(0.01, 0.99, 99)
      - "a:b:c"：np.arange(a,b,c)（含端点修正）
      - "0.1,0.2,0.25,0.33"：逗号列表
    """
    if not grid_spec or str(grid_spec).strip().lower() == "auto":
        return np.linspace(0.01, 0.99, 99)

    s = str(grid_spec).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError("threshold_grid 形如 '0.1:0.9:0.01' 或逗号列表")
        a, b, c = map(float, parts)
        arr = np.arange(a, b + 1e-12, c)
        arr = arr[(arr > 0) & (arr < 1)]
        return np.unique(np.clip(arr, 1e-6, 1-1e-6))

    vals = [float(x) for x in s.split(",") if x.strip()]
    vals = [v for v in vals if 0 < v < 1]
    if not vals:
        raise ValueError("threshold_grid 中没有合法阈值（0~1 之间）")
    return np.array(sorted(set(vals)))


def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str, grid: np.ndarray):
    """
    仅二分类：在给定阈值网格上对指定 metric 进行搜索。
    返回：(best_thr, best_score, default@0.5, [(thr, score), ...])
    支持的 metric: f1, f1_macro, recall, precision, accuracy, youden_j
    """
    metric = (metric or "").lower()
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim > 1:
        # 取正类概率（约定第 2 列），若只有一列也直接用
        if y_prob.shape[1] >= 2:
            y_prob = y_prob[:, 1]
        else:
            y_prob = y_prob.ravel()

    def _score_at_thr(t: float) -> float:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            return f1_score(y_true, y_pred, average="binary")
        elif metric == "f1_macro":
            return f1_score(y_true, y_pred, average="macro")
        elif metric == "recall":
            return classification_report(y_true, y_pred, output_dict=True)["weighted avg"]["recall"]
        elif metric == "precision":
            return classification_report(y_true, y_pred, output_dict=True)["weighted avg"]["precision"]
        elif metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif metric in ("youden", "youden_j", "j"):
            # TPR + TNR - 1
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn + 1e-12)
            tnr = tn / (tn + fp + 1e-12)
            return float(tpr + tnr - 1.0)
        else:
            # 默认 F1
            return f1_score(y_true, y_pred, average="binary")

    series = []
    best_thr, best_score = None, -1.0
    for t in grid:
        s = _score_at_thr(t)
        series.append((float(t), float(s)))
        if s > best_score:
            best_score, best_thr = s, float(t)

    default_score = _score_at_thr(0.5)
    return float(best_thr), float(best_score), float(default_score), series


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
    # ——（第 6 步）样本/不平衡
    sample_weight: Optional[np.ndarray] = None,
    class_weight: Optional[str] = None,   # "balanced" or None
    # ——（第 7 步）阈值调优
    optimize_metric: Optional[str] = None,
    threshold_grid: Optional[str] = "auto",
) -> Tuple[Pipeline, dict]:

    model_params = model_params or {}
    y = df[target]
    X = df.drop(columns=[target])

    task = task_override or _infer_task(y)
    report: Dict = {"task": task}

    # —— 样本权重 & class_weight=balanced（仅分类）
    sw = None
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype="float64")
        if sw.shape[0] != len(df):
            raise ValueError("sample_weight 长度必须与 df 行数一致")
    if task == "classification" and class_weight and class_weight.lower() == "balanced":
        # 计算类权重
        classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        n_classes = len(classes)
        cw = {c: total / (n_classes * cnt) for c, cnt in zip(classes, counts)}
        cw_vec = np.vectorize(lambda v: cw.get(v, 1.0))(y.values)
        sw = cw_vec if sw is None else sw * cw_vec
        report["class_weight"] = {"balanced": cw}

    # —— 划分训练/测试
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None if task == "regression" else y
    )
    sw_tr = None
    if sw is not None:
        # 对应训练集索引
        sw_tr = sw[y.index.get_indexer(Xtr.index)]

    # —— 预处理
    pre = _build_preprocessor(Xtr, target=target, enable=preprocess)
    # —— 构建模型
    model = get_model(model_name, task, **model_params)

    # ============ CV 搜索（不与早停混用） ============
    if isinstance(cv, int) and cv >= 2:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        pipe = build_pipeline(pre, model)

        # 处理 param_grid
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
        if sw_tr is not None:
            fit_kwargs["model__sample_weight"] = sw_tr  # 管道最末模型接收

        searcher.fit(Xtr, ytr, **fit_kwargs)
        best_pipe = searcher.best_estimator_
        yhat = best_pipe.predict(Xte)

        if task == "classification":
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
            report["_y_true"], report["_y_prob"], report["_y_pred"] = yte.tolist(), (yprob.tolist() if yprob is not None else None), yhat.tolist()
        else:
            rmse = float(mean_squared_error(yte, yhat, squared=False))
            r2 = float(r2_score(yte, yhat))
            report.update({"rmse": rmse, "r2": r2})

        # 阈值调优（仅二分类）
        if task == "classification" and optimize_metric and report.get("_y_prob") is not None and len(np.unique(yte)) == 2:
            try:
                grid = _parse_threshold_grid(threshold_grid)
                best_thr, best_score, default_score, series = _tune_threshold(np.asarray(yte), np.asarray(report["_y_prob"]), optimize_metric, grid)
                report["threshold_tuning"] = {
                    "metric": optimize_metric,
                    "best_threshold": float(best_thr),
                    "best_score": float(best_score),
                    "default_threshold_score": float(default_score),
                }
                report["threshold_curve"] = [{"thr": float(t), "score": float(s)} for t, s in series]
                print(f"🎯 阈值调优：metric={optimize_metric} | best_thr={best_thr:.3f} | best={best_score:.4f} | default@0.5={default_score:.4f}")
            except Exception as e:
                print(f"⚠️ 阈值调优失败：{e}")

        # 完整返回（含 best_params/best_score）
        return best_pipe, {**report, "cv": True, "best_params": getattr(searcher, "best_params_", None),
                           "best_score": getattr(searcher, "best_score_", None)}

    # ============ 非 CV：支持早停 ============
    if early_stopping and model_name in ("lgbm", "xgb"):
        Xtr2, Xval, ytr2, yval = train_test_split(
            Xtr, ytr, test_size=val_size, random_state=random_state, stratify=None if task == "regression" else ytr
        )
        pre = _build_preprocessor(Xtr, target=target, enable=preprocess)  # 用 X 构建
        Xtr2_t = pre.fit_transform(Xtr2, ytr2) if preprocess else Xtr2.values
        Xval_t = pre.transform(Xval) if preprocess else Xval.values
        Xte_t  = pre.transform(Xte)  if preprocess else Xte.values

        fit_kwargs = {}
        if model_name == "lgbm":
            import lightgbm as lgb
            fit_kwargs["eval_set"] = [(Xval_t, yval)]
            if eval_metric: fit_kwargs["eval_metric"] = eval_metric
            fit_kwargs["callbacks"] = [lgb.early_stopping(es_rounds, verbose=False)]
            if sw_tr is not None:
                fit_kwargs["sample_weight"] = sw_tr[ytr.index.get_indexer(Xtr2.index)]
        elif model_name == "xgb":
            fit_kwargs["eval_set"] = [(Xval_t, yval)]
            if eval_metric: fit_kwargs["eval_metric"] = eval_metric
            fit_kwargs["early_stopping_rounds"] = es_rounds
            if sw_tr is not None:
                fit_kwargs["sample_weight"] = sw_tr[ytr.index.get_indexer(Xtr2.index)]
            if hasattr(model, "get_xgb_params") and "verbose" not in model.get_xgb_params():
                fit_kwargs["verbose"] = False

        clf = model
        clf.fit(Xtr2_t, ytr2, **fit_kwargs)

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

        if task == "classification":
            acc = float(accuracy_score(yte, yhat))
            f1m = float(f1_score(yte, yhat, average="macro"))
            report.update({"accuracy": acc, "f1_macro": f1m})
            if yprob is not None and len(np.unique(yte)) == 2:
                report["roc_auc"] = float(roc_auc_score(yte, yprob))
            report["classification_report"] = classification_report(yte, yhat, output_dict=True)
            report["_y_true"], report["_y_prob"], report["_y_pred"] = yte.tolist(), (yprob.tolist() if yprob is not None else None), yhat.tolist()
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

        # 阈值调优
        if task == "classification" and optimize_metric and report.get("_y_prob") is not None and len(np.unique(yte)) == 2:
            try:
                grid = _parse_threshold_grid(threshold_grid)
                best_thr, best_score, default_score, series = _tune_threshold(np.asarray(yte), np.asarray(report["_y_prob"]), optimize_metric, grid)
                report["threshold_tuning"] = {
                    "metric": optimize_metric,
                    "best_threshold": float(best_thr),
                    "best_score": float(best_score),
                    "default_threshold_score": float(default_score),
                }
                report["threshold_curve"] = [{"thr": float(t), "score": float(s)} for t, s in series]
                print(f"🎯 阈值调优：metric={optimize_metric} | best_thr={best_thr:.3f} | best={best_score:.4f} | default@0.5={default_score:.4f}")
            except Exception as e:
                print(f"⚠️ 阈值调优失败：{e}")

        pipe = build_pipeline(pre, clf)
        return pipe, report

    # ============ 普通训练（无早停） ============
    pipe = build_pipeline(pre, model)
    fit_kwargs = {}
    if sw_tr is not None:
        fit_kwargs["model__sample_weight"] = sw_tr
    pipe.fit(Xtr, ytr, **fit_kwargs)

    yhat = pipe.predict(Xte)
    if task == "classification":
        yprob = None
        try:
            proba = pipe.predict_proba(Xte)
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
        report["_y_true"], report["_y_prob"], report["_y_pred"] = yte.tolist(), (yprob.tolist() if yprob is not None else None), yhat.tolist()
    else:
        rmse = float(mean_squared_error(yte, yhat, squared=False))
        r2 = float(r2_score(yte, yhat))
        report.update({"rmse": rmse, "r2": r2})

    # 阈值调优
    if task == "classification" and optimize_metric and report.get("_y_prob") is not None and len(np.unique(yte)) == 2:
        try:
            grid = _parse_threshold_grid(threshold_grid)
            best_thr, best_score, default_score, series = _tune_threshold(np.asarray(yte), np.asarray(report["_y_prob"]), optimize_metric, grid)
            report["threshold_tuning"] = {
                "metric": optimize_metric,
                "best_threshold": float(best_thr),
                "best_score": float(best_score),
                "default_threshold_score": float(default_score),
            }
            report["threshold_curve"] = [{"thr": float(t), "score": float(s)} for t, s in series]
            print(f"🎯 阈值调优：metric={optimize_metric} | best_thr={best_thr:.3f} | best={best_score:.4f} | default@0.5={default_score:.4f}")
        except Exception as e:
            print(f"⚠️ 阈值调优失败：{e}")

    # 过拟合/可疑提示
    if task == "classification" and report.get("accuracy") == 1.0 and report.get("f1_macro") == 1.0:
        report["warning_perfect_score"] = "训练/验证得分为 1.0，请留意是否过拟合或数据泄漏（也可能是数据可分性极强的演示集）。"

    return pipe, report
