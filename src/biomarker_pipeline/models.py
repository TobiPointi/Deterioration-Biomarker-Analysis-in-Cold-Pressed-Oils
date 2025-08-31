from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

@dataclass
class ModelSettings:
    lasso_alpha: float = 0.001
    rf_n_estimators: int = 1000
    rf_random_state: int = 42

def run_lasso(X: pd.DataFrame, y: np.ndarray, alpha: float) -> pd.Series:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.T)  # samples × features
    model = Lasso(alpha=alpha, max_iter=20000, random_state=42)
    model.fit(Xs, y)
    coefs = pd.Series(model.coef_, index=X.index, name="lasso_coef")
    return coefs

def run_random_forest(X: pd.DataFrame, y: np.ndarray, n_estimators: int, random_state: int) -> pd.Series:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True,
        max_features="sqrt"
    )
    rf.fit(X.T, y)
    imp = pd.Series(rf.feature_importances_, index=X.index, name="rf_importance")
    return imp

def run_spearman(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rhos = []
    pvals = []
    for feat in X.index:
        rho, pval = spearmanr(X.loc[feat].values, y)
        rhos.append(rho)
        pvals.append(pval)
    out = pd.DataFrame({"spearman_rho": rhos, "spearman_pval": pvals}, index=X.index)
    out["direction"] = np.where(out["spearman_rho"] >= 0, "increase", "decrease")
    return out
