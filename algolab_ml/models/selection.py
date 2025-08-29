from __future__ import annotations
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

def cv_score(model, X, y, task: str="classification", cv: int=5, scoring: str=None, random_state: int=42):
    if scoring is None:
        scoring = "roc_auc" if task=="classification" and len(set(y))==2 else "accuracy" if task=="classification" else "r2"
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state) if task=="classification" else KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, scoring=scoring, cv=kf)
    return {"scoring": scoring, "mean": float(np.mean(scores)), "std": float(np.std(scores)), "scores": scores.tolist()}
