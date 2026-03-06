# Changelog

All notable changes to scOPE are documented here.

---

## [0.2.0] — 2026-03-05

### Performance
- `scope/decomposition/base.py` — `_get_X` now returns a sparse matrix when the input is sparse instead of unconditionally calling `.toarray()`; added `_get_X_dense` for callers (e.g. `fit`) that genuinely require a dense array; sparse @ dense matmul in `transform` avoids ever materialising a full dense cell × gene matrix for large sc datasets
- `scope/pipeline/sc_pipeline.py` — `SingleCellPipeline.transform` now preserves sparsity throughout steps 2–4: gene subsetting uses sparse column slicing, zero-padding uses COO construction, and densification only occurs once immediately before the decomposer projection; previously `.toarray()` was called on the full unsubsetted matrix (e.g. 200k × 26k genes, ~20 GB), causing kernel crashes on large datasets

---

## [0.1.8] — 2026-03-05

### Fixed
- `scope/evaluation/svd_evaluation.py` — removed leftover unused `import matplotlib` after Agg removal, fixing ruff F401 lint error

---

## [0.1.7] — 2026-03-05

### Fixed
- `scope/evaluation/svd_evaluation.py` — removed `matplotlib.use("Agg")` backend override that was silently clobbering the matplotlib backend for the entire process when the module was imported, suppressing interactive plot display in Jupyter notebooks

---

## [0.1.6] — 2026-03-05

### Fixed
- `scope/evaluation/svd_evaluation.py` — restored missing imports (`numpy`, `pandas`, `seaborn`, `scipy`, `sklearn`) that were accidentally dropped during refactoring in v0.1.5

---

## [0.1.5] — 2026-03-05

### Fixed
- `scope/evaluation/svd_evaluation.py` — replaced `matplotlib.use("Agg")` module-level call with a conditional check (`DISPLAY` env var / OS) to avoid overriding the backend in interactive environments; added `os` import

---

## [0.1.4] — 2026-03-05

### Fixed
- `scope/evaluation/svd_evaluation.py` — `SVDEvaluator._extract_coef` now
  correctly unwraps `_SklearnWrapper` (scope's internal classifier wrapper)
  before extracting `coef_`, fixing `AttributeError` on initialization

---

## [0.1.3] — 2026-03-05

### Fixed
- `scope/decomposition/cnmf.py` — renamed unused loop variable `i` → `_i` (ruff B007)
- `scope/evaluation/svd_evaluation.py` — removed unused imports (`warnings`, `Optional`, `Sequence`, `roc_curve`); removed unused `bars` variable assignment; replaced `try/except/pass` with `contextlib.suppress`; rewrote `dict()` call as a dict literal; fixed import sort order (ruff F401, F841, SIM105, C408, I001, UP035)
- `scope/preprocessing/bulk.py` — removed unused `List` and `Optional` imports; replaced `import scanpy as _sc` availability check with `importlib.util.find_spec`; fixed yoda condition; scanpy now imported inline where used (ruff F401, UP035, SIM300)
- `scope/preprocessing/single_cell.py` — removed unused `Optional` import (ruff F401)

---

## [0.1.2] — 2026-03-05

### Added

**Decomposition**
- `FactorAnalysisDecomposition` (`decomposition="fa"`) — probabilistic factor analysis via sklearn; models gene-specific noise variance (heteroscedasticity), making it more appropriate than PCA/SVD when technical noise varies widely across genes
- `ConsensusNMFDecomposition` (`decomposition="cnmf"`) — consensus NMF following Kotliar et al. (eLife 2019); runs NMF `n_iter` times with random initialisations, clusters the resulting components via K-means, and refits with the consensus H matrix fixed; produces substantially more stable and reproducible gene programs than single-run NMF
- Both new decompositions are registered in `get_decomposition()` and usable as drop-in replacements via the `decomposition=` argument in `BulkPipeline`

**Evaluation**
- `SVDEvaluator` — comprehensive SVD component interpretation class; takes a fitted `BulkPipeline` and bulk latent embedding and produces 15 diagnostic plots + a tidy gene program CSV:
  - Classifier-weighted scree plot (`|coef|×σ` colour-coded bars)
  - Component importance bar chart (ranked; decomposed `|coef|` vs σ)
  - Hierarchically clustered gene loading heatmap
  - Per-component signed gene bar charts
  - Pairwise latent scatter coloured by mutation label
  - Component score separation violins (Mann-Whitney p-values)
  - Spearman component–label correlation heatmap (FDR-corrected)
  - Cross-validated ROC curve + component ablation AUC curve
  - Permutation test (observed AUROC vs. null distribution)
  - Gene biplot (sample scores + gene loading arrows)
  - SHAP dot and bar summaries (optional; requires `shap`)
  - UMAP of Z_bulk coloured by mutation label (optional; requires `umap-learn`)
  - Calibration curve (reliability of predicted probabilities)
  - Component cross-correlation heatmap (flags batch bleed)
  - `gene_program_table.csv` — tidy export of component rank, σ, `|coef|`, importance, and top gene loadings

**Preprocessing — `BulkPreprocessor`**
- `already_log_transformed` flag — skips the log1p step when input is already log-normalised (e.g. log2-TPM from GEO)
- `min_samples_expressed` / `min_expression` — removes lowly expressed genes before fitting
- `gene_blacklist` — unconditionally excludes specified gene symbols
- `auto_remove_mito` — removes `MT-` / `mt-` prefixed genes
- `auto_remove_ribo` — removes `RPS` / `RPL` ribosomal protein genes
- `run_hvg` / `n_hvg` / `hvg_flavor` — highly variable gene selection via scanpy (falls back to top-variance selection if scanpy unavailable)
- `batch_key` / `batch_method` — batch correction via ComBat (`pip install combat`) or Harmony (`pip install harmonypy`)
- All new parameters have defaults that preserve v0.1.1 behavior; fully backward compatible

**Preprocessing — `SingleCellPreprocessor`**
- `max_genes` — upper bound on detected genes per cell (doublet proxy, complements `max_counts`)
- `max_mito_pct` — removes cells exceeding a mitochondrial fraction threshold
- `auto_flag_mito` — annotates `pct_mito` in `adata.obs` even when mito filtering is disabled
- `run_doublet_detection` / `doublet_threshold` — Scrublet-based doublet scoring and removal (optional; requires `pip install scrublet`)
- `already_qc_filtered` — skips all cell-level QC filters for pre-filtered inputs
- `already_normalized` — skips library-size normalisation
- `already_log_transformed` — skips log1p
- All new parameters have defaults that preserve v0.1.1 behavior; fully backward compatible

**Documentation**
- README updated to document all new decomposition methods, preprocessing options, SVDEvaluator usage, output file reference table, and optional dependency table

---

## [0.1.1] — 2026-03-04

### Added
- Full package codebase groundwork (`scope/` Python package, `pyproject.toml`, `MANIFEST.in`, `LICENSE`)
- `BulkPipeline` — end-to-end Phase 1: bulk normalisation → SVD/NMF/ICA/PCA decomposition → per-mutation classifier training with optional cross-validation
- `SingleCellPipeline` — end-to-end Phase 2: sc QC/normalisation → gene alignment → bulk latent projection → per-cell mutation probability inference
- `BulkPreprocessor` / `BulkNormalizer` / `BulkScaler` — bulk normalisation (CPM, TPM, median-ratio, TMM) and gene-wise centering/scaling
- `SingleCellPreprocessor` — sc QC filtering and library-size normalisation
- `BulkSCAligner` — bulk–sc distribution alignment (z-score, moment-matching, quantile)
- `SVDDecomposition`, `NMFDecomposition`, `ICADecomposition`, `PCADecomposition` with shared `fit` / `transform` / `components_` interface
- `PerMutationClassifierSet` with logistic, random forest, GBM, XGBoost, LightGBM, SVM, and MLP classifiers
- `evaluate_classifier`, `evaluate_all`, `cross_validate_classifiers`, `roc_curve_data`, `pr_curve_data`
- `compute_umap`, `compute_tsne`, `plot_embedding`, `plot_mutation_probabilities`, `plot_scree`, `plot_mutation_heatmap`
- PyPI release (`pip install scope-bio`) and conda-forge recipe
- Initial notebooks and test suite

---

## [0.1.0] — 2026-03-03

### Added
- Initial repository structure, README, and proof-of-concept notebooks
