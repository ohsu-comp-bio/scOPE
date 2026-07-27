# scOPE — single-cell Oncological Prediction Explorer

[![PyPI version](https://img.shields.io/pypi/v/scope-bio?v=0.2.0)](https://pypi.org/project/scope-bio/)
[![pypi downloads](https://img.shields.io/pepy/dt/scope-bio?label=pypi%20downloads)](https://pepy.tech/project/scope-bio)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/scope-bio?cacheSeconds=300)](https://anaconda.org/conda-forge/scope-bio)
[![conda-forge downloads](https://img.shields.io/conda/dn/conda-forge/scope-bio?label=conda-forge%20downloads&cacheSeconds=300)](https://anaconda.org/conda-forge/scope-bio)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

**Transfer driver-associated transcriptional programs from genotype-rich bulk tumors into single-cell RNA-seq—without refitting the target cohort.**

Bulk tumor cohorts provide matched genotype and expression across hundreds of patients, but collapse every cell into one profile. Single-cell RNA-seq resolves cellular heterogeneity, but usually cannot observe somatic genotype reliably in each cell. **scOPE bridges those complementary measurement regimes.** For each cancer type, it learns driver-associated expression structure in bulk RNA-seq, freezes the fitted feature map and driver models, and projects gene-matched single cells through that same map.

> [!IMPORTANT]
> **scOPE does not call mutant alleles.** A scOPE score ranks cells by similarity to a bulk-derived, driver-associated transcriptional program. It is evidence about phenotype—not proof that an individual cell carries the mutation.

<picture>
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/Ashford-A/scOPE/main/assets/figures/scOPE_overview_dark.png">
  <img src="https://raw.githubusercontent.com/Ashford-A/scOPE/main/assets/figures/scOPE_overview_light.png"
       alt="scOPE bulk-to-single-cell transfer-learning framework"
       width="100%">
</picture>

---

## Framework at a glance

scOPE separates **learning**, **transfer**, and **interpretation**:

1. **Learn in bulk.** Build a cancer-specific latent representation from bulk RNA-seq with matched driver labels, then train one supervised model per driver.
2. **Freeze the transfer map.** Preserve the fitted gene order, preprocessing parameters, latent loadings, and classifier parameters.
3. **Project single cells.** Gene-match and align scRNA-seq to the bulk reference, then map every cell into the fixed bulk-derived latent space—without using single-cell mutation labels or refitting the latent axes.
4. **Score driver-associated programs.** Apply each frozen driver model to each projected cell to obtain a continuous program score.
5. **Decide whether to trust the transfer.** The manuscript framework evaluates held-out bulk performance, label-free transfer confidence, direct mutation-transcript support where available, patient-level aggregation, longitudinal behavior, cell-state localization, CNV concordance, and negative controls.

The central question is therefore not *“Can expression predict every mutation?”* It is:

> **Which driver-associated expression programs are reproducible enough in bulk to survive transfer into single cells—and where do those programs localize once transferred?**

Across the manuscript audit, transfer was empirically selective: 158 driver–cancer models were attempted across seven malignancies, 102 met predefined claim-safety criteria, and only 11 reached out-of-fold AUROC ≥ 0.90. scOPE is designed to expose that selectivity rather than hide it.

---

## Mathematical formulation

### Notation

| Symbol | Meaning |
|---|---|
| `c` | cancer type |
| `d` | driver gene or alteration |
| `i` | tumor sample or single cell, depending on context |
| `k` | latent dimensionality; the manuscript analysis used `k = 30` |
| `V_c,k` | frozen bulk-derived gene-loading matrix |
| `β_d`, `α_d` | frozen driver-specific classifier parameters |

### 1. Learn a cancer-specific representation in bulk

For cancer type `c`, the bulk expression matrix contains `n_c` tumors and `p_c` genes:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equations/bulk_shape_dark.png">
    <img src="assets/equations/bulk_shape_light.png" alt="X bulk c is an n c by p c real-valued matrix" width="500">
  </picture>
</p>

After fitting the bulk preprocessing transform, scOPE factorizes the standardized matrix with a rank-`k` truncated SVD:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equations/bulk_svd_dark.png">
    <img src="assets/equations/bulk_svd_light.png" alt="Standardized bulk expression is approximated by U Sigma V transpose" width="760">
  </picture>
</p>

The tumor-level latent representation is obtained by projecting onto the bulk gene-loading axes:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equations/bulk_latent_dark.png">
    <img src="assets/equations/bulk_latent_light.png" alt="Z bulk equals standardized X bulk times V and equals U Sigma" width="760">
  </picture>
</p>

`V_c,k` contains the **bulk-derived gene-loading axes**, while `Z_bulk,c` is the **tumor-by-factor representation**. Each factor is a weighted multigene expression axis rather than a single-gene marker.

### 2. Learn one driver model in the bulk latent space

For each driver `d`, the default logistic model learns a multicomponent expression program from matched tumor-level mutation labels:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equations/bulk_model_dark.png">
    <img src="assets/equations/bulk_model_light.png" alt="Bulk driver probability equals sigmoid of intercept plus latent factors times driver coefficients" width="1000">
  </picture>
</p>

Bulk performance is evaluated out of fold. Preprocessing, SVD, and driver fitting are repeated within each training fold so held-out tumors do not influence their own predictions.

### 3. Transfer the frozen representation into single cells

For the same cancer type, let the single-cell matrix contain `m_c` cells and `p_sc,c` measured genes:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equations/sc_shape_dark.png">
    <img src="assets/equations/sc_shape_light.png" alt="X single cell c is an m c by p single cell c real-valued matrix" width="520">
  </picture>
</p>

scOPE gene-matches the target cohort to the fitted bulk feature space and applies the selected label-free alignment transform. The aligned cells are then projected through the **unchanged bulk loadings**:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equations/sc_projection_dark.png">
    <img src="assets/equations/sc_projection_light.png" alt="Z single cell equals aligned X single cell times the frozen bulk loading matrix V" width="680">
  </picture>
</p>

The **unchanged bulk driver classifier** is then applied to every projected cell:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equations/sc_score_dark.png">
    <img src="assets/equations/sc_score_light.png" alt="Cell driver-program score equals sigmoid of the frozen bulk classifier applied to the projected cell" width="830">
  </picture>
</p>

This fixed reuse of `V_c,k`, `alpha_d`, and `beta_d` is the transfer-learning step. The target cells enter the bulk-derived coordinate system; the coordinate system is not relearned around the single-cell cohort. The canonical analysis used label-free moment matching while leaving the bulk SVD loadings and classifier parameters fixed.

### 4. Interpret the score correctly

The raw score `s_i,d` answers:

> **How strongly does cell `i` express the bulk-derived transcriptional program associated with driver `d`?**

It is **not automatically** the probability that the cell carries the mutation. Bulk mutation labels can also encode lineage, tumor purity, molecular subtype, co-mutation, or cohort composition. scOPE therefore uses an evidence ladder and supports abstention when transfer evidence is weak.

### 5. Manuscript interpretation layers

<details>
<summary><strong>Cell-state residualization</strong></summary>

Raw transferred scores can be elevated in healthy cells when a bulk program captures lineage structure shared across assays. The manuscript subtracts the matched healthy-reference median for the cell's canonical state:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equations/residual_dark.png">
    <img src="assets/equations/residual_light.png" alt="Residual score equals raw score minus the median score among matched healthy reference cells in the same cell state" width="1000">
  </picture>
</p>

When fewer than 25 same-state reference cells are available, the cohort-wide reference median is used instead. A positive residual indicates program activity above the matched reference baseline; it still does not constitute an allele call.

</details>

<details>
<summary><strong>Ground-truth-free transfer confidence</strong></summary>

The manuscript combines four label-free properties: bulk transferability (`T`), spatial coherence (`S`), score concentration (`C`), and agreement with an expression-derived CNV axis (`X`).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equations/confidence_t_dark.png">
    <img src="assets/equations/confidence_t_light.png" alt="Transferability component derived from bulk AUROC" width="900">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equations/confidence_sc_dark.png">
    <img src="assets/equations/confidence_sc_light.png" alt="Spatial coherence and score concentration components" width="900">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equations/confidence_x_dark.png">
    <img src="assets/equations/confidence_x_light.png" alt="CNV agreement component derived from CNV AUROC" width="820">
  </picture>
</p>

Available components are combined with a weighted geometric mean:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equations/confidence_composite_dark.png">
    <img src="assets/equations/confidence_composite_light.png" alt="Composite confidence is a weighted geometric mean of available evidence components" width="1000">
  </picture>
</p>

The manuscript weights are `w_T = 1`, `w_S = 1.5`, `w_C = 1`, and `w_X = 1`. Confidence is a **triage and abstention signal**, not a posterior genotype probability.

</details>

---

## What scOPE gives you

- A fitted, reusable bulk-to-single-cell projection for each cancer type.
- One continuous driver-program score per cell and driver.
- Interpretable latent factors and signed gene loadings.
- Cross-validated bulk metrics, calibration and permutation diagnostics, component ablations, SHAP summaries, and gene-program tables.
- A natural interface for comparing transferred programs with cell state, treatment time, spatial context, mutation-transcript evidence, or inferred CNV.

---

## Installation

### From PyPI
```bash
pip install scope-bio
```

### With optional dependencies (UMAP, XGBoost, LightGBM, SHAP)
```bash
pip install scope-bio[full]
```

### From conda-forge
```bash
conda install -c conda-forge scope-bio
```

### Development install
```bash
git clone https://github.com/Ashford-A/scOPE.git
cd scOPE
conda env create -f environments/scope-dev.yml
conda activate scope-dev
pip install -e ".[dev]"
```

---

## Quick start

```python
import anndata as ad
import pandas as pd
from scope import BulkPipeline, SingleCellPipeline
from scope.io import load_mutation_labels

# --- Phase 1: Bulk --------------------------------------------------------
adata_bulk = ad.read_h5ad("bulk_cohort.h5ad")
mutation_labels = load_mutation_labels("mutations.csv", sample_col="sample_id")

bulk_pipe = BulkPipeline(
    norm_method="cpm",
    decomposition="svd",   # "svd" | "nmf" | "ica" | "pca" | "fa" | "cnmf"
    n_components=30,       # paper used 30; user-configurable
    classifier="logistic",    # "logistic" | "random_forest" | "gbm" | "xgboost" | "lightgbm" | "svm" | "mlp"
)
bulk_pipe.fit(adata_bulk, mutation_labels, cv=5)
bulk_pipe.save("models/bulk_pipeline.pkl")

# --- Phase 2: Single cell --------------------------------------------------
adata_sc = ad.read_h5ad("sc_tumor.h5ad")

adata_bulk_pp = bulk_pipe.preprocessor_.transform(adata_bulk)

sc_pipe = SingleCellPipeline(
    bulk_pipeline=bulk_pipe,
    alignment_method="moment_matching",  # alignment used in the paper
)
sc_pipe.fit(adata_bulk_pp, adata_sc)
adata_sc = sc_pipe.transform(adata_sc)

# Historical API names: mutation_prob_KRAS, mutation_prob_TP53, ...
# Interpret these as raw driver-program scores, not direct allele probabilities.

# --- Visualise -------------------------------------------------------------
from scope.visualization import compute_umap, plot_mutation_probabilities

adata_sc = compute_umap(adata_sc, obsm_key="X_svd")
fig = plot_mutation_probabilities(adata_sc, mutations=["KRAS", "TP53"])
fig.savefig("mutation_probs.pdf", bbox_inches="tight")
```

---

## Preprocessing options

`BulkPreprocessor` and `SingleCellPreprocessor` are designed to handle the full range of input states — from raw counts to already-normalized matrices — without requiring you to re-implement preprocessing outside the pipeline.

### Bulk: handling already-processed inputs

```python
bulk_pipe = BulkPipeline(
    norm_method="none",               # skip library-size normalization
    log1p=False,                      # data is already log-transformed
    decomposition="svd",
    n_components=50,
)

# Or pass flags directly to BulkPreprocessor:
from scope.preprocessing import BulkPreprocessor

preprocessor = BulkPreprocessor(
    norm_method="cpm",
    log1p=True,
    already_log_transformed=False,    # set True if input is e.g. log2-TPM from GEO
    min_samples_expressed=5,          # remove genes expressed in fewer than 5 samples
    min_expression=0.5,               # expression threshold for the above filter
    gene_blacklist=["MALAT1", "NEAT1"],
    auto_remove_mito=True,            # remove MT- genes
    auto_remove_ribo=True,            # remove RPS/RPL genes
    run_hvg=True,                     # select highly variable genes (requires scanpy)
    n_hvg=3000,
    hvg_flavor="seurat_v3",
    batch_key="cohort",               # batch correction via obs column
    batch_method="combat",            # "combat" | "harmony"
)
```

### Single-cell: QC, mito filtering, and doublet removal

```python
from scope.preprocessing import SingleCellPreprocessor

sc_prep = SingleCellPreprocessor(
    filter_strategy="both",           # "min_counts" | "min_genes" | "both" | "none"
    min_counts=500,
    min_genes=300,
    max_counts=25000,
    max_genes=6000,                   # upper bound (doublet proxy)
    max_mito_pct=20.0,                # remove cells with >20% mitochondrial reads
    auto_flag_mito=True,              # annotate pct_mito in adata.obs regardless
    run_doublet_detection=True,       # Scrublet-based doublet removal (pip install scrublet)
    doublet_threshold=None,           # None = automatic Scrublet threshold
    already_qc_filtered=False,        # True = skip all cell-level filters
    already_normalized=False,         # True = skip library-size normalization
    already_log_transformed=False,    # True = skip log1p
)
```

---

## Decomposition methods

scOPE supports several latent-space methods, all sharing the same `fit` / `transform` / `components_` interface and usable as a drop-in via the `decomposition=` argument in `BulkPipeline`.

| Key | Class | Notes |
|---|---|---|
| `"svd"` | `SVDDecomposition` | Default. Linear, interpretable. Gene loadings V enable direct sc projection. |
| `"nmf"` | `NMFDecomposition` | Non-negative, additive gene programs (metagenes). Requires non-negative input. |
| `"ica"` | `ICADecomposition` | Independent components. Useful for finding non-Gaussian expression sources. |
| `"pca"` | `PCADecomposition` | Standard PCA. Equivalent to SVD on centred data. |
| `"fa"` | `FactorAnalysisDecomposition` | Probabilistic FA. Accounts for gene-specific noise variance (heteroscedasticity). |
| `"cnmf"` | `ConsensusNMFDecomposition` | Repeated NMF followed by component clustering for more stable additive gene programs. |

```python
# Consensus NMF example
bulk_pipe = BulkPipeline(
    decomposition="cnmf",
    decomposition_kwargs={"n_iter": 50, "n_components": 20},
    classifier="logistic",
)

# Factor Analysis example
bulk_pipe = BulkPipeline(
    decomposition="fa",
    n_components=30,
    classifier="logistic",
)
```

---

## SVD evaluation

When using `decomposition="svd"`, `SVDEvaluator` produces a comprehensive suite of plots and a gene-program table that show which latent components drive a classifier and which genes define those components. These outputs support model auditing and biological interpretation; they do not by themselves establish driver specificity.

```python
from scope.evaluation import SVDEvaluator

# After fitting:
adata_pp = bulk_pipe.transform_bulk(adata_bulk)
Z_bulk = adata_pp.obsm[bulk_pipe.obsm_key_]

ev = SVDEvaluator(bulk_pipe, Z_bulk, mutation="KRAS")
ev.run_all(output_dir="figures/svd_eval_KRAS")
```

`run_all()` saves the following to `output_dir/`:

| Output | Description |
|---|---|
| `weighted_scree.png` | Scree plot with bars colour-coded by `\|coef\|×σ` classifier importance |
| `component_importance.png` | Ranked bar chart of importance; decomposed `\|coef\|` vs σ |
| `gene_loading_heatmap.png` | Hierarchically clustered heatmap of top gene loadings × top components |
| `top_genes_per_component.png` | Signed gene bar charts for each top-important component |
| `latent_scatter.png` | Pairwise scatter of top components coloured by mutation label |
| `separation_violins.png` | Component score distributions by mutation status + Mann-Whitney p |
| `component_label_correlation.png` | Spearman ρ heatmap between components and mutation label (FDR-corrected) |
| `roc_ablation.png` | Cross-validated ROC curve + AUC vs. n_components retained |
| `permutation_test.png` | Observed AUROC vs. permutation null distribution |
| `gene_biplot.png` | Sample scores + gene loading arrows for top-2 components |
| `shap_summary_dot.png` | SHAP dot summary (requires `shap`) |
| `shap_summary_bar.png` | SHAP bar summary (requires `shap`) |
| `umap_zbulk.png` | UMAP of Z_bulk coloured by mutation label (requires `umap-learn`) |
| `calibration_curve.png` | Reliability diagram for predicted mutation probabilities |
| `component_crosscorr.png` | Pearson correlation among SVD components (flags batch bleed) |
| `gene_program_table.csv` | Tidy table: component rank, σ, `\|coef\|`, importance, top genes + loadings |

Individual plots can also be called directly:

```python
ev.plot_weighted_scree(output_dir=Path("figures/"))
ev.plot_separation_violins(output_dir=Path("figures/"), top_components=12)
ev.export_gene_program_table(output_dir=Path("figures/"), top_components=10)
```

---

## API reference

### Preprocessing
| Class | Description |
|---|---|
| `BulkNormalizer` | CPM / TPM / median-ratio / TMM normalisation |
| `BulkScaler` | Gene-wise centering and scaling |
| `BulkPreprocessor` | Combined normalise + scale, with gene filtering, HVG, and batch correction |
| `SingleCellPreprocessor` | QC filter + mito filter + doublet removal + normalise + optional scale |
| `BulkSCAligner` | z-score / moment-matching / quantile alignment |

### Decomposition
| Class | Description |
|---|---|
| `SVDDecomposition` | Truncated SVD (randomized / ARPACK / full) |
| `NMFDecomposition` | Non-negative matrix factorization |
| `ICADecomposition` | FastICA |
| `PCADecomposition` | PCA via sklearn |
| `FactorAnalysisDecomposition` | Probabilistic factor analysis (heteroscedastic noise) |
| `ConsensusNMFDecomposition` | Consensus NMF for stable gene program discovery |
| `get_decomposition(name)` | Factory function |

### Classification
| Class | Description |
|---|---|
| `LogisticMutationClassifier` | L1/L2/ElasticNet logistic regression |
| `RandomForestMutationClassifier` | Random forest |
| `GBMMutationClassifier` | Gradient boosting (sklearn) |
| `XGBMutationClassifier` | XGBoost |
| `LGBMMutationClassifier` | LightGBM |
| `SVMMutationClassifier` | SVM + Platt calibration |
| `MLPMutationClassifier` | Multi-layer perceptron |
| `PerMutationClassifierSet` | Trains/stores one classifier per mutation |
| `get_classifier(name)` | Factory function |

### Evaluation
| Class / Function | Description |
|---|---|
| `SVDEvaluator` | Full SVD component interpretation suite (15 plots + gene program CSV) |
| `evaluate_classifier` | AUROC, AUPRC, Brier score |
| `evaluate_all` | Evaluate all mutations at once |
| `cross_validate_classifiers` | Stratified k-fold CV |
| `roc_curve_data` / `pr_curve_data` | Curve arrays for plotting |

### Visualization
| Function | Description |
|---|---|
| `compute_umap` | UMAP on latent embedding |
| `compute_tsne` | t-SNE on latent embedding |
| `plot_embedding` | Scatter by categorical or continuous |
| `plot_mutation_probabilities` | Grid of per-driver program-score overlays |
| `plot_scree` | Singular value / EVR scree plot |
| `plot_mutation_heatmap` | Mean probability per cluster heatmap |

---

## Optional dependencies

| Package | Purpose | Install |
|---|---|---|
| `umap-learn` | UMAP embeddings and SVDEvaluator UMAP plot | `pip install scope-bio[full]` |
| `shap` | SHAP component importance in SVDEvaluator | `pip install scope-bio[full]` |
| `xgboost` | XGBoost classifier | `pip install scope-bio[full]` |
| `lightgbm` | LightGBM classifier | `pip install scope-bio[full]` |
| `scrublet` | Doublet detection in SingleCellPreprocessor | `pip install scrublet` |
| `combat` | ComBat batch correction in BulkPreprocessor | `pip install combat` |
| `harmonypy` | Harmony batch correction in BulkPreprocessor | `pip install harmonypy` |
| `statsmodels` | FDR correction in SVDEvaluator correlation heatmap | `pip install statsmodels` |

---

## Citation

The scOPE preprint is available on bioRxiv:

> Ashford, A. J., Lapadat, A., & Demir, E. (2026). **scOPE identifies which driver-associated expression programs transfer from bulk tumors to single cells.** *bioRxiv*. https://doi.org/10.64898/2026.07.24.740598

If you use scOPE in your research, please cite:

```bibtex
@article{Ashford2026scOPE,
  title   = {{scOPE} identifies which driver-associated expression programs transfer from bulk tumors to single cells},
  author  = {Ashford, Andrew J. and Lapadat, Alex and Demir, Emek},
  journal = {bioRxiv},
  year    = {2026},
  month   = jul,
  doi     = {10.64898/2026.07.24.740598},
  url     = {https://doi.org/10.64898/2026.07.24.740598},
  note    = {Preprint posted July 26, 2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
