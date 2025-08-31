
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional

def minmax_01(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    vmin, vmax = s.min(), s.max()
    if np.isclose(vmax, vmin):
        return pd.Series(0.5, index=s.index)  # flat -> neutral score
    return (s - vmin) / (vmax - vmin)

def to_positive_importance(series: pd.Series) -> pd.Series:
    # Take absolute values so both up/down trends can be "important"
    return series.abs()

def combine_modalities(
    mod_tables: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    method_weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """Merge per-modality tables (index=feature) to a single frame with normalized scores and final score.
    Expected columns per modality: lasso_coef, rf_importance, spearman_rho.
    method_weights: dict with keys in {lasso, rf, correlation}, values sum to 1.0 (optional).
    """
    # Default method weights if none given
    mw = {"lasso": 1/3, "rf": 1/3, "correlation": 1/3}
    if method_weights:
        # normalize to sum=1 across provided keys
        keys = ["lasso", "rf", "correlation"]
        tmp = {k: float(method_weights.get(k, 0.0)) for k in keys}
        s = sum(tmp.values()) or 1.0
        mw = {k: v/s for k, v in tmp.items()}

    scaled = {}
    for name, df in mod_tables.items():
        # Extract raw columns (may be missing)
        lasso_raw = df.get("lasso_coef", pd.Series(dtype=float))
        rf_raw = df.get("rf_importance", pd.Series(dtype=float))
        rho_raw = df.get("spearman_rho", pd.Series(dtype=float))

        # Convert to positive importances
        lasso = to_positive_importance(lasso_raw)
        rf = to_positive_importance(rf_raw)
        rho = to_positive_importance(rho_raw)

        # Minmax scale within modality
        lasso_s = minmax_01(lasso)
        rf_s = minmax_01(rf)
        rho_s = minmax_01(rho)

        # Weighted mean per-modality score across methods
        # Drop missing per row and renormalize weights dynamically
        df_s = pd.concat([lasso_s, rf_s, rho_s], axis=1)
        df_s.columns = [f"{name}_lasso", f"{name}_rf", f"{name}_rho"]

        method_map = {
            f"{name}_lasso": mw["lasso"],
            f"{name}_rf": mw["rf"],
            f"{name}_rho": mw["correlation"],
        }

        def weighted_row_mean(row: pd.Series) -> float:
            present = row.dropna()
            if present.empty:
                return np.nan
            w = np.array([method_map[c] for c in present.index], dtype=float)
            w = w / w.sum()
            return float(np.dot(present.values, w))

        df_s[f"{name}_score01"] = df_s.apply(weighted_row_mean, axis=1)
        scaled[name] = df_s

    # Outer-join all modalities
    merged = None
    for name, tbl in scaled.items():
        merged = tbl if merged is None else merged.join(tbl, how="outer")

    # Weighted average across modalities (renormalize to present per row)
    score_cols = [f"{k}_score01" for k in scaled.keys()]
    weights_series = pd.Series(weights, dtype=float)

    def weighted_modal_mean(row: pd.Series) -> float:
        present = row[score_cols].dropna()
        if present.empty:
            return np.nan
        w = weights_series.reindex([c.replace("_score01","") for c in present.index])
        w = w / w.sum()
        return float(np.dot(present.values, w.values))

    merged["FinalScore01"] = merged.apply(weighted_modal_mean, axis=1)
    merged["FinalScore"] = merged["FinalScore01"] * 100.0
    return merged.sort_values("FinalScore", ascending=False)
