from __future__ import annotations
import pandas as pd
from ..data.io import detect_problem_type, train_test_split_df
from ..features.transform import build_tabular_preprocess
from .model_zoo import get_model

def fit_tabular(df: pd.DataFrame, target: str, model_name: str="xgb", test_size: float=0.2, random_state: int=42, preprocess: bool=True, model_params: dict | None=None):
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error

    Xtr, Xte, ytr, yte = train_test_split_df(df, target, test_size=test_size, random_state=random_state)
    task = detect_problem_type(ytr)
    model_params = model_params or {}
    model = get_model(model_name, task=task, **model_params)
    steps = []
    if preprocess:
        pre = build_tabular_preprocess(Xtr)
        if pre: steps.append(("pre", pre))
    steps.append(("model", model))
    pipe = Pipeline(steps)
    pipe.fit(Xtr, ytr)

    preds = pipe.predict(Xte)
    report = {"task": task}
    if task == "classification":
        from sklearn.metrics import classification_report
        report["accuracy"] = float(accuracy_score(yte, preds))
        report["f1_macro"] = float(f1_score(yte, preds, average="macro"))
        if len(set(ytr)) == 2:
            try:
                proba = pipe.predict_proba(Xte)[:,1]
                report["roc_auc"] = float(roc_auc_score(yte, proba))
            except Exception:
                pass
        report["classification_report"] = classification_report(yte, preds, output_dict=True)
    else:
        report["rmse"] = float(mean_squared_error(yte, preds, squared=False))
        report["r2"]   = float(r2_score(yte, preds))
        report["mae"]  = float(mean_absolute_error(yte, preds))
    return pipe, report
