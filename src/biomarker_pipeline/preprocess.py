from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Optional

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean(axis=1).values[:, None]) / (df.std(axis=1, ddof=0).replace(0, np.nan).values[:, None])

def clr(df: pd.DataFrame, epsilon: float = 1e-9) -> pd.DataFrame:
    """Centered log-ratio transform row-wise (features as rows)."""
    arr = df.to_numpy(dtype=float)
    arr = np.where(arr <= 0, epsilon, arr)
    gm = np.exp(np.mean(np.log(arr), axis=1, keepdims=True))
    log_ratio = np.log(arr / gm)
    out = pd.DataFrame(log_ratio, index=df.index, columns=df.columns)
    return out

def combine_quantified_unquantified(
    time_mat: pd.DataFrame,
    quantified_mask: Optional[pd.Series] = None,
    assume_quantified: bool = False
) -> pd.DataFrame:
    if assume_quantified:
        return zscore(time_mat)

    if quantified_mask is None:
        quantified_mask = pd.Series(False, index=time_mat.index)

    q = time_mat.loc[quantified_mask]
    uq = time_mat.loc[~quantified_mask]

    out_parts = []
    if not q.empty:
        out_parts.append(zscore(q))
    if not uq.empty:
        out_parts.append(zscore(clr(uq)))
    if out_parts:
        return pd.concat(out_parts, axis=0)
    # fallback
    return zscore(time_mat)
