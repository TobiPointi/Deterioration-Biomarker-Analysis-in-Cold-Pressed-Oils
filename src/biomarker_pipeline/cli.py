from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import pandas as pd
import numpy as np

from . import __version__
from .io import load_table, select_value_columns, enforce_index, extract_time_matrix
from .preprocess import combine_quantified_unquantified
from .models import ModelSettings, run_lasso, run_random_forest, run_spearman
from .pca_plot import pca_2d_feature_scatter
from .scoring import combine_modalities

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.sort_index().to_csv(path)

def modality_pipeline(
    name: str,
    file: str,
    time_map: Dict[str, float],
    out_dir: Path,
    index_column: Optional[str] = None,
    value_selector: Optional[str] = None,
    assume_quantified: bool = False,
    quantified_list: Optional[List[str]] = None,
    lasso_alpha: float = 0.001,
    rf_n_estimators: int = 1000,
    rf_random_state: int = 42,
    fig_dpi: int = 220,
    fig_format: str = "png",
) -> Dict[str, pd.DataFrame]:
    _ensure_dir(out_dir)
    raw = load_table(file)

    # Set index and pick value columns if requested
    if index_column is not None:
        raw = enforce_index(raw, index_column=index_column)

    # value selection (e.g., 'Mean' columns in GC-MS)
    val_df = select_value_columns(raw, value_selector=value_selector)

    # Subset columns to timepoints in desired order
    time_mat, ordered_labels, days = extract_time_matrix(val_df, time_map)
    time_mat = time_mat.apply(pd.to_numeric, errors="coerce")

    # Build quantified mask
    quantified_mask = None
    if not assume_quantified and quantified_list is not None:
        quantified_mask = time_mat.index.isin(quantified_list)

    # Preprocess (zscore for quantified; CLR+z for unquantified)
    proc = combine_quantified_unquantified(
        time_mat, quantified_mask=quantified_mask, assume_quantified=assume_quantified
    )

    # Save processed
    proc_out = out_dir / "processed_matrix.parquet"
    proc.to_parquet(proc_out)

    # Modeling
    lasso = run_lasso(proc, days, alpha=lasso_alpha)
    rf = run_random_forest(proc, days, n_estimators=rf_n_estimators, random_state=rf_random_state)
    spr = run_spearman(proc, days)

    # Save tables
    _write_csv(lasso.to_frame("lasso_coef"), out_dir / "lasso_coefficients.csv")
    _write_csv(rf.to_frame("rf_importance"), out_dir / "random_forest_importance.csv")
    _write_csv(spr, out_dir / "spearman.csv")

    # PCA figure
    pca_2d_feature_scatter(proc, f"PCA ({name})", out_dir / "pca_scores", dpi=fig_dpi, fmt=fig_format)

    # Return for aggregation
    merged = lasso.to_frame().join(rf.to_frame(), how="outer").join(spr, how="outer")
    return {name: merged}

def main():
    parser = argparse.ArgumentParser(description="Unified oil biomarker pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    time_map = cfg["timepoints"]
    results_dir = Path(cfg.get("results_dir", "results"))
    _ensure_dir(results_dir)

    modeling = cfg.get("modeling", {})
    lasso_alpha = float(modeling.get("lasso_alpha", 0.001))
    rf_n_estimators = int(modeling.get("rf_n_estimators", 1000))
    rf_random_state = int(modeling.get("rf_random_state", 42))

    figures = cfg.get("figures", {})
    fig_dpi = int(figures.get("dpi", 220))
    fig_format = str(figures.get("format", "png"))

    combined: Dict[str, pd.DataFrame] = {}
    for m in cfg["modalities"]:
        name = m["name"]
        out_dir = results_dir / name
        combined.update(
            modality_pipeline(
                name=name,
                file=m["file"],
                time_map=time_map,
                out_dir=out_dir,
                index_column=m.get("index_column"),
                value_selector=m.get("value_selector"),
                assume_quantified=bool(m.get("assume_quantified", False)),
                quantified_list=m.get("quantified_list"),
                lasso_alpha=lasso_alpha,
                rf_n_estimators=rf_n_estimators,
                rf_random_state=rf_random_state,
                fig_dpi=fig_dpi,
                fig_format=fig_format,
            )
        )

    # Aggregate across modalities
    weights = cfg.get("aggregation", {}).get("weights", {})
    min_modalities_required = int(cfg.get("aggregation", {}).get("min_modalities_required", 1))
    top_n = int(cfg.get("aggregation", {}).get("top_n", 30))

    method_weights = cfg.get('aggregation', {}).get('method_weights', None)
    combined_tbl = combine_modalities(combined, weights=weights, method_weights=method_weights)
    # Drop rows that don't meet min modality presence
    score_cols = [f"{k}_score01" for k in combined.keys()]
    present_modalities = (~combined_tbl[score_cols].isna()).sum(axis=1)
    combined_tbl = combined_tbl.loc[present_modalities >= min_modalities_required]

    combined_dir = results_dir / "combined"
    _ensure_dir(combined_dir)
    combined_tbl.to_csv(combined_dir / "combined_scores.csv")

    top_tbl = combined_tbl.head(top_n)
    top_tbl.to_csv(combined_dir / "top_biomarkers.csv")

    print(f"[OK] Pipeline finished. Results in: {results_dir.resolve()}")

if __name__ == "__main__":
    main()
