# Deterioration-Biomarker-Analysis-in-Cold-Pressed-Oils

to identify deterioration biomarkers in cold-pressed oils across:

- **Polyphenols (LC–MS/MS)**
- **Fatty acids (GC–FID)**
- **Secondary lipid oxidation products (GC–MS)**

It standardizes preprocessing (including compositional handling), runs **PCA**, **LASSO**, **Random Forest**, and **rank-based correlations** over storage time, and then **combines modality-specific scores** into a robust, **missing-aware final ranking**.

> Built for longitudinal T0–T9 designs (e.g., days `[0, 3, 7, 14, 28, 42, 56, 84, 112, 168]`) but configurable for any schedule.

---

## Quick Start

1. **Install** (ideally in a fresh virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure** your paths and settings by copying and editing:
   ```bash
   cp configs/example_config.yaml configs/my_config.yaml
   ```

3. **Run the pipeline**:
   ```bash
   python -m biomarker_pipeline.cli --config configs/my_config.yaml
   ```

Outputs (tables and figures) are written to `results/` by default.

---

## Input Expectations

Each modality input is a **wide table** with **features as rows** and **time points as columns**.

- If your GC–MS file contains both `Mean` and `SD` columns (e.g., `T0 Mean`, `T0 SD`), specify `value_selector: 'Mean'` in the config.
- You can provide a list of **quantified** features for GC–MS; quantified get **z-scored**, unquantified are treated as **compositional** (CLR + z-score).
- LC–MS and GC–FID are usually fully quantified; set `assume_quantified: true` or define `quantified_list: []` accordingly.

You can supply Excel (`.xlsx`) or CSV (`.csv`) files.

---

## Methods, Reproducibility, and Choices

- **Scaling**: quantified → z-score; unquantified → CLR then z-score
- **Compositional**: CLR uses a small epsilon to avoid log(0)
- **Models**:
  - **LASSO (regression)** to predict actual day values (continuous target)
  - **Random Forest Regressor** with `n_estimators=1000`, `random_state=42`, `n_jobs=-1`
  - **Spearman rho** vs time (and p-value)
- **Aggregation**:
  - Per-modality scores are min–max scaled to `[0, 1]` (higher is better)
  - Missing scores are **ignored, not zeroed**
  - Final score = weighted mean of available modality scores (weights from config)
- **PCA**: done on standardized features; top-2 components plotted; explained variance reported
- **Clustering**: optional hierarchical clustering on processed features
- **Determinism**: all stochastic steps have fixed seeds

All parameters are configurable via YAML.

---

## Outputs

- `results/{modality}/`:
  - `processed_matrix.parquet`: processed feature × time matrix
  - `lasso_coefficients.csv`
  - `random_forest_importance.csv`
  - `spearman.csv` (`rho`, `pval`, `direction`)
  - `pca_explained_variance.json`
  - `pca_scores.png` / `.pdf` (feature scatter)
- `results/combined/`:
  - `combined_scores.csv` – merged per-modality results with a **FinalScore (0–100)**
  - `top_biomarkers.csv` – top-N by final score (configurable)

---

## Config Reference

See `configs/example_config.yaml` for an annotated example. Key items:

- `timepoints`: mapping of column names (e.g., `T0`, `T1`, …) to numeric days
- `modalities`: list of modality blocks with:
  - `name`: `polyphenols` | `fatty_acids` | `gc_ms` (free text allowed)
  - `file`: path to `.xlsx`/`.csv`
  - `index_column`: feature ID/name column if not already row index
  - `value_selector`: substring to select value columns (e.g., `'Mean'`)
  - `assume_quantified`: bool (treat all as quantified)
  - `quantified_list`: optional list of quantified features
- `aggregation`: weights for each modality, `top_n` cutoff, and minimum modalities required

---

## Citing

If this pipeline aids your publication, please cite using `CITATION.cff`.


### Intra-method weighting (your ranking approach)
You can weight the three methods **within each modality** to mirror your ranking code:

```yaml
aggregation:
  method_weights:
    lasso: 0.4
    rf: 0.4
    correlation: 0.2
```
