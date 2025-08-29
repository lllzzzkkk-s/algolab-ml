from __future__ import annotations
from typing import List, Tuple
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from .model_zoo import get_model

def soft_voting(task: str, base_models: List[Tuple[str,str]]):
    estimators = [(name, get_model(alias, task=task)) for name, alias in base_models]
    return VotingClassifier(estimators=estimators, voting="soft") if task=="classification" else VotingRegressor(estimators=estimators)

def stacking(task: str, base_models: List[Tuple[str,str]], final_model_alias: str):
    estimators = [(name, get_model(alias, task=task)) for name, alias in base_models]
    final_est = get_model(final_model_alias, task=task)
    return StackingClassifier(estimators=estimators, final_estimator=final_est) if task=="classification" else StackingRegressor(estimators=estimators, final_estimator=final_est)
