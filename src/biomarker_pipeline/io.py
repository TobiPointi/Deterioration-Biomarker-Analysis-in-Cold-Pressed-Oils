from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Union

def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)

def select_value_columns(df: pd.DataFrame, value_selector: Optional[str]) -> pd.DataFrame:
    if value_selector is None:
        return df
    cols = [c for c in df.columns if value_selector.lower() in str(c).lower()]
    if not cols:
        raise ValueError(f"No columns matched value_selector='{value_selector}'. Available: {list(df.columns)}")
    return df[cols]

def enforce_index(df: pd.DataFrame, index_column: Optional[str]) -> pd.DataFrame:
    if index_column is None:
        if df.index.name is None and isinstance(df.index, pd.RangeIndex):
            raise ValueError("Features are not indexed. Provide index_column in config.")
        return df
    out = df.copy()
    if index_column not in out.columns:
        raise ValueError(f"index_column '{index_column}' not found in columns.")
    out = out.set_index(index_column)
    return out

def extract_time_matrix(df: pd.DataFrame, time_map: Dict[str, float]) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """Return matrix with columns ordered by provided time_map keys, preserving feature index.

    - Picks columns whose names match keys in time_map
    - Returns (matrix, ordered_labels, numeric_days)
    """
    available = {c: c for c in df.columns}
    chosen = []
    days = []
    for label, day in time_map.items():
        # allow relaxed matching if exact not present
        matches = [c for c in df.columns if str(c).strip().lower() == str(label).strip().lower()]
        if not matches:
            # try startswith (e.g., 'T0 Mean' -> 'T0')
            matches = [c for c in df.columns if str(c).strip().lower().startswith(str(label).strip().lower())]
        if not matches:
            continue
        chosen.append(matches[0])
        days.append(day)
    if not chosen:
        raise ValueError("None of the specified timepoint labels were found in the table.")
    mat = df[chosen].copy()
    return mat, chosen, np.array(days, dtype=float)
