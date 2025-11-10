# src/wrappers.py
import numpy as np
import xgboost as xgb

def _predict_proba_booster(booster: xgb.Booster, X):
    dm = xgb.DMatrix(X)
    best_ntree_limit = getattr(booster, "best_ntree_limit", None)
    best_iteration  = getattr(booster, "best_iteration", None)
    if best_ntree_limit is not None:
        p = booster.predict(dm, ntree_limit=best_ntree_limit)
    elif best_iteration is not None:
        p = booster.predict(dm, iteration_range=(0, best_iteration + 1))
    else:
        p = booster.predict(dm)
    p = np.asarray(p).reshape(-1)
    return np.c_[1 - p, p]

class XGBBoosterWithThreshold:
    """Wrapper để pickle an toàn: giữ booster + threshold."""
    def __init__(self, booster: xgb.Booster, threshold: float = 0.5):
        self._booster = booster
        self.threshold = float(threshold)

    def get_booster(self) -> xgb.Booster:
        return self._booster

    def predict_proba(self, X):
        return _predict_proba_booster(self._booster, X)

    def predict(self, X, threshold=None):
        th = self.threshold if threshold is None else float(threshold)
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= th).astype(int)
