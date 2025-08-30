from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def plot_roc_pr_curves(y_true, y_prob, out_dir: Path):
    try:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        if y_prob.ndim > 1 and y_prob.shape[1] >= 2:
            y_prob = y_prob[:, 1]
        # ROC
        RocCurveDisplay.from_predictions(y_true, y_prob)
        _savefig(Path(out_dir) / "plots" / "roc_curve.png")
        # PR
        PrecisionRecallDisplay.from_predictions(y_true, y_prob)
        _savefig(Path(out_dir) / "plots" / "pr_curve.png")
        print("✅ 已保存 ROC/PR 曲线")
    except Exception:
        pass

def plot_feature_importance(pipeline, feature_names, out_dir: Path, top_n: int = 30):
    # 获取末端模型
    try:
        model = pipeline.named_steps.get("model", None)
    except Exception:
        model = None
    if model is None:
        return
    # 支持 feature_importances_
    if not hasattr(model, "feature_importances_"):
        return
    importances = np.asarray(model.feature_importances_)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]
    # 取前 top_n
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
    vals = importances[idx]
    plt.figure(figsize=(8, max(4, top_n*0.25)))
    y = np.arange(len(idx))
    plt.barh(y, vals)
    plt.yticks(y, names)
    plt.gca().invert_yaxis()
    plt.xlabel("importance")
    plt.title("Top Feature Importances")
    _savefig(Path(out_dir) / "plots" / "feature_importance.png")
    print("✅ 已保存特征重要性")

def save_cv_results(estimator, out_dir: Path):
    # 对 GridSearchCV/RandomizedSearchCV 的 best_estimator_ 才有 cv_results_
    try:
        cv_results_ = getattr(estimator, "cv_results_", None)
        if cv_results_:
            import pandas as pd
            df = pd.DataFrame(cv_results_)
            p = Path(out_dir) / "cv" / "cv_results.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)
            print(f"✅ 已保存 CV 结果：{p.resolve()}")
    except Exception:
        pass

def plot_learning_curve(evals_result: dict, out_dir: Path, metric_hint: str | None = None, task: str | None = None):
    """
    ev for LightGBM: {'training': {'auc':[...], ...}, 'valid_0': {'auc':[...], ...}}
    ev for XGBoost : {'validation_0': {'auc':[...], ...}, 'validation_1': {...}}
    """
    try:
        # 选 metric
        # 优先 eval_metric；否则按任务默认
        metric = (metric_hint or ("auc" if task == "classification" else "rmse")).lower()
        # 取两个系列（train / valid）
        # LightGBM 风格
        train_key = next((k for k in evals_result.keys() if "train" in k or "training" in k), None)
        valid_key = next((k for k in evals_result.keys() if "valid" in k), None)
        # XGBoost 风格
        if train_key is None:
            train_key = next((k for k in evals_result.keys() if "validation_1" in k), None)  # 有时 val1 作为 train 接近线
        if valid_key is None:
            valid_key = next((k for k in evals_result.keys() if "validation_0" in k or "valid_0" in k), None)
        if valid_key is None and len(evals_result) == 1:
            valid_key = list(evals_result.keys())[0]

        if valid_key is None:
            return

        # 取曲线
        v = evals_result[valid_key]
        if metric not in v:
            # 尝试找到一个最像的 metric
            if len(v) > 0:
                metric = list(v.keys())[0]
        series = {}
        if train_key and metric in evals_result.get(train_key, {}):
            series["train"] = evals_result[train_key][metric]
        if metric in evals_result[valid_key]:
            series["valid"] = evals_result[valid_key][metric]

        if not series:
            return

        # 画图
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        for name, arr in series.items():
            plt.plot(arr, label=name)
        plt.xlabel("iteration")
        plt.ylabel(metric)
        plt.title(f"Learning Curve ({metric})")
        plt.legend()
        _savefig(Path(out_dir) / "plots" / "learning_curve.png")
        print("✅ 已保存学习曲线")
    except Exception:
        pass

def plot_confusion_matrices(y_true, y_prob, out_dir, thresholds: dict | None = None):
    """保存默认0.5与最佳F1阈值下的混淆矩阵图（仅二分类）"""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim != 1:
        return  # 多分类概率不绘制

    pairs = [("cm_default_0.5.png", 0.5)]
    if isinstance(thresholds, dict) and "best_f1" in thresholds:
        pairs.append(("cm_best_f1.png", float(thresholds["best_f1"]["threshold"])))

    for fname, thr in pairs:
        y_pred = (y_prob >= thr).astype(int)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, values_format="d")
        ax.set_title(f"Confusion Matrix @ threshold={thr:.3f}")
        fig.tight_layout()
        fig.savefig(str(Path(out_dir) / fname), dpi=150)
        plt.close(fig)

def plot_calibration_curve(y_true, y_prob, out_dir, n_bins: int = 10):
    """概率校准曲线（仅二分类）"""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibrationDisplay

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim != 1:
        return

    fig, ax = plt.subplots(figsize=(4.5, 4))
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=n_bins, ax=ax)
    ax.set_title("Calibration Curve")
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "calibration_curve.png"), dpi=150)
    plt.close(fig)
