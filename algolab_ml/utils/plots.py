from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd

def _ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def plot_roc_pr_curves(y_true, y_prob, out_dir: Path):
    out_dir = Path(out_dir); _ensure_dir(out_dir)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        prob_pos = y_prob
    else:
        prob_pos = y_prob[:, 1]  # 正类
    # ROC
    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0,1], [0,1], lw=1, linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=160)
    plt.close()
    # PR
    precision, recall, _ = precision_recall_curve(y_true, prob_pos)
    ap = average_precision_score(y_true, prob_pos)
    plt.figure()
    plt.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=160)
    plt.close()

def plot_feature_importance(estimator, feature_names, out_dir: Path, top_n: int = 30):
    out_dir = Path(out_dir); _ensure_dir(out_dir)
    # 拿到最终模型
    model = getattr(estimator, "best_estimator_", estimator)
    if hasattr(model, "named_steps"):  # Pipeline
        model = model.named_steps.get("model", model)
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return
    imp = np.asarray(imp)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(imp))]
    order = np.argsort(-imp)[:top_n]
    plt.figure(figsize=(8, max(4, int(0.35 * len(order)))))
    plt.barh(range(len(order)), imp[order])
    plt.yticks(range(len(order)), [str(feature_names[i]) for i in order])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (top %d)" % len(order))
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=160)
    plt.close()

def save_cv_results(estimator, out_dir: Path):
    out_dir = Path(out_dir); _ensure_dir(out_dir)
    est = getattr(estimator, "best_estimator_", estimator)
    if hasattr(estimator, "cv_results_"):
        df = pd.DataFrame(estimator.cv_results_)
        df.to_csv(out_dir / "cv_results.csv", index=False)
