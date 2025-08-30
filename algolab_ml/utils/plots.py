from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, confusion_matrix

def save_roc_pr(y_true, y_score, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_predictions(y_true, y_score)
    fig = plt.gcf()
    fig.savefig(Path(out_dir) / "roc_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    fig = plt.gcf()
    fig.savefig(Path(out_dir) / "pr_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_confusion(y_true, y_pred, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    fig.savefig(Path(out_dir) / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
