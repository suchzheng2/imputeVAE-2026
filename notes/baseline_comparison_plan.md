# Baseline Comparison Plan

**Motivation:** ISMB reviewers (R1, R4, R5, R6) all flagged missing baselines as a major weakness. This is the #1 action item for the paper.

## Baselines to implement

### Tier 1 — Must have (for poster/paper)

| # | Method | Type | Why | Effort |
|---|--------|------|-----|--------|
| 1 | **Linear regression** | Simple | Already done (SUC-8). Per-gene linreg on CT means. | Done |
| 2 | **scVI** | VAE | Gold standard. Reviewers explicitly asked for it. Treat cell/nucleus as batch, use `transform_batch` for translation. | Medium |
| 3 | **Harmony + kNN** | Integration | Reviewers asked for batch-correction baseline. Show that batch correction alone doesn't produce gene-level translation. | Low |
| 4 | **Cell-type mean** | Trivial | Map each nucleus to its cell-type mean expression. Floor baseline. | Low |

### Tier 2 — Strong additions (for paper)

| # | Method | Type | Why | Effort |
|---|--------|------|-----|--------|
| 5 | **scGen** | Perturbation VAE | Latent arithmetic (global shift vector). | Medium — needs separate env (Python 3.10, scvi-tools 0.17.x) |
| 6 | **scANVI** | Supervised VAE | Uses cell-type labels. Semi-supervised scVI. | Low (once scVI works) |

### Tier 3 — Aspirational (if time permits)

| # | Method | Type | Why | Effort |
|---|--------|------|-----|--------|
| 7 | **moscot** | Optimal transport | pip installable, Python 3.12 compatible, actively maintained (v0.5.1, Jun 2026). Uses Fused Gromov-Wasserstein. | Medium |
| 8 | **scGPT** | Foundation model | Impressive if included, but complex setup. | High |

### Dropped

| Method | Reason |
|--------|--------|
| **CellOT** | Unmaintained since 2022, Python 3.9 only, GitHub-clone only. Use moscot instead. |

## Compatibility Notes (from research 2026-07-08)

| Method | Python | Install | GPU | Maintained |
|--------|--------|---------|-----|------------|
| scVI (scvi-tools) | 3.12 | `pip install scvi-tools` (v1.4.3) | Yes (PyTorch) | Active |
| scANVI | 3.12 | Part of scvi-tools | Yes | Active |
| scGen | **3.10** | `pip install scgen==2.1.0 scvi-tools==0.17.4` | Yes | **Broken with scvi-tools>=1.0** |
| Harmony | 3.12 | `pip install harmonypy` | No (CPU) | Active |
| moscot | 3.12 | `pip install moscot` (v0.5.1) | Yes (JAX) | Active |

## Comparison protocol

### Data setup (same for all methods)
- Gene list: `data/training_gene_list_v3.csv` (10,134 genes, Config B)
- Cell data: `10X_cells_v2_v3_AIBS/`, batch `c3A` (v3 protocol)
- Nuclei data: `10X_nuclei_v2_v3_AIBS/`, batch `n3A` (v3 protocol)
- Train/test split: 80/20, stratified by cell type (same split as our VAE)
- Common cell types only (19 types from `data/common_cell_types.csv`)
- Normalization: log1p for MSE-based methods, raw counts for NB-based

### Translation task
- **Input:** nucleus expression profile (test set)
- **Output:** predicted cell expression profile
- **Ground truth:** real cell expression (test set, matched by cell type)

### Evaluation metrics

Three tiers, designed to address reviewer concerns about circular evaluation (R1, R5).

**Tier 1 — CT-level (does translation get the right direction?)**

| Metric | What it measures | Concern |
|--------|-----------------|---------|
| Gene correlation (CT means) | Mean expression profile accuracy | CT-mean baseline scores well here too |
| R-squared (CT means) | Variance explained at CT level | Same concern |
| Shift score | Did imputed move toward cell, away from nucleus? | Key differentiator |
| Classification accuracy | Correct CT assignment? | R1 flagged as circular — use but don't rely on |
| DE overlap (top 100) | Marker gene preservation | Downstream-relevant |
| Coexpression | Gene-gene correlation structure | Tests biological relationships |
| Gene MSE (CT means) | Raw reconstruction error | Simple, interpretable |

**Tier 2 — Sample-level (does it preserve per-cell biology?)**

| Metric | What it measures | Why it matters |
|--------|-----------------|---------------|
| Variance ratio | var(imputed) / var(real cell) within CT | <<1 = variance collapse, key VAE weakness |
| Within-CT Spearman | Per-cell ranking preserved? | Detects mean-hallucination |
| Per-cell corr to CT mean | How close to mean? | Inverse proxy for variance retention |

**Tier 3 — Label-free (addresses R1/R5 circularity concern)**

| Metric | What it measures | Why label-free matters |
|--------|-----------------|----------------------|
| QQ gene correlation | Quantile-quantile distribution shape match per gene | No CT labels needed |
| Wasserstein distance | Distribution distance per gene (top 500 variable) | OT-based, no labels |

**Tier 4 — Biological validation (the strongest differentiator)**

| Metric | What it measures | Why it kills scVI |
|--------|-----------------|-------------------|
| Delta vs ground truth (r, R2) | Correlation of predicted delta (imputed - nucleus) with real delta (cell - nucleus), per gene per CT | Tests if method learns the *correct transformation*, not just removes batch |
| Ribosomal gene sign | Do ribosomal proteins (RPL/RPS) increase? | Should be +delta (cytoplasmic, lost in nuc isolation). Batch correction would flatten. |
| Nuclear lncRNA sign | Do MALAT1/Xist/NEAT1 decrease? | Should be -delta (nuclear-enriched). Batch correction would flatten. |
| Cytoplasmic gene sign | Do Actb/Calm1/Vamp2/14-3-3 increase? | Same logic as ribosomal |
| CT-specific delta | Do neuronal markers (Snap25, Syt1, Nrgn) increase more in neurons than glia? | Tests cell-type-specific transformation. scGen's global shift can't do this. |

**Why this tier matters most for the paper:**
- scVI treats modality as batch noise → removes differences → delta ≈ 0 for everything
- scGen applies a single global shift → same delta for all cell types
- Linear regression applies per-gene slopes → can get direction but not CT-specificity
- Our cVAE learns CT-conditioned, gene-specific, directional transformation → gets biology right
- GO enrichment on our delta already confirmed: ribosome (p=3e-26), cytoplasm (p=8e-08), synapse (p=3e-14)

**Script:** `analysis/evaluate_baseline.py` — unified eval for any method.
Outputs JSON to `{output_dir}/eval_{name}.json`.

### Fair comparison notes
- **scVI/scANVI:** Train on combined cell+nucleus, use `get_normalized_expression(transform_batch="cell")` on nucleus input to get translated profiles
- **Harmony:** After integration, use kNN in corrected space to find nearest cell neighbors for each nucleus, average their expression
- **scGen:** Train on cell+nucleus with cell_type and modality labels, use `predict(ctrl_key="nucleus", stim_key="cell", celltype_to_predict=CT)` per cell type
- **Cell-type mean:** For each nucleus, assign its cell-type mean cell expression (trivial but important floor)

## Implementation

### 1. scVI baseline

**Install:** `pip install scvi-tools` (into impute2 env)

**Approach:** Strategy A — train on combined data with modality as batch covariate.

```python
import scvi
import anndata as ad

# Combine cell and nucleus data
adata = ad.concat([cell_adata, nuc_adata], label="modality", keys=["cell", "nucleus"])
adata.layers["counts"] = adata.X.copy()  # raw counts

# Setup and train
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="modality")
model = scvi.model.SCVI(adata, n_latent=30, n_layers=2)
model.train(max_epochs=200, early_stopping=True)

# Translate: what would each nucleus look like as a cell?
# get_normalized_expression with transform_batch generates counterfactual
nuc_mask = adata.obs["modality"] == "nucleus"
imputed = model.get_normalized_expression(
    adata[nuc_mask],
    transform_batch="cell",  # decode as if from cell batch
    library_size=1e4,        # normalize to same depth
)
# imputed is a DataFrame (cells x genes) — the predicted cell expression
imputed_log1p = np.log1p(imputed.values)
```

**Key parameters:**
- `n_latent=30` (match our VAE's latent dim)
- `n_layers=2`
- `max_epochs=200` with early stopping
- `gene_likelihood="nb"` (default, negative binomial)

### 2. scGen baseline

**Install:** `pip install scgen` (separate package, uses scvi-tools backend)

**Approach:** Perturbation prediction — learns a latent shift vector between nucleus and cell.

```python
import scgen

# Need cell_type and modality in obs
adata.obs["condition"] = adata.obs["modality"]  # "cell" or "nucleus"

# Setup and train
scgen.SCGEN.setup_anndata(adata, batch_key="condition")
model = scgen.SCGEN(adata)
model.train(max_epochs=100, batch_size=128, early_stopping=True)

# Predict per cell type
predictions = []
for ct in cell_types:
    pred, delta = model.predict(
        ctrl_key="nucleus",
        stim_key="cell",
        celltype_to_predict=ct,
        obs_key="cell_type",
    )
    predictions.append(pred)
imputed = ad.concat(predictions)
```

**Key notes:**
- Does NOT require paired data (perfect for our setting)
- Predictions are per cell type — applies same shift to all cells of a CT
- This means within-CT variation comes from the original nucleus variation + a constant shift
- Should preserve per-cell variation better than our VAE (but shift is crude)

### 3. Harmony + kNN baseline

**Install:** `pip install harmonypy`

**Approach:** Integrate in PCA space, then use kNN to transfer expression.

```python
import harmonypy
from sklearn.neighbors import NearestNeighbors

# PCA on combined data
sc.tl.pca(adata, n_comps=50)

# Harmony correction
ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, "modality")
adata.obsm["X_harmony"] = ho.Z_corr.T

# For each nucleus, find k nearest cell neighbors in Harmony space
cell_mask = adata.obs["modality"] == "cell"
nuc_mask = adata.obs["modality"] == "nucleus"

nn = NearestNeighbors(n_neighbors=20).fit(adata[cell_mask].obsm["X_harmony"])
dists, idxs = nn.kneighbors(adata[nuc_mask].obsm["X_harmony"])

# Average neighbor cell expression as prediction
cell_X = adata[cell_mask].X
imputed = np.array([cell_X[idx].mean(axis=0) for idx in idxs])
```

**Key notes:**
- Harmony only corrects the embedding, doesn't generate expression — we use kNN transfer
- kNN averaging will partially collapse variance (by design)
- This is what reviewers asked for: show that batch correction ≠ translation

### 4. Cell-type mean baseline

**Approach:** Trivially assign each nucleus the mean cell expression of its cell type.

```python
ct_means = {}
for ct in cell_types:
    ct_means[ct] = cell_X[cell_adata.obs["cell_type"] == ct].mean(axis=0)

imputed = np.array([ct_means[ct] for ct in nuc_adata.obs["cell_type"]])
```

**Key notes:**
- Variance ratio = 0 by definition (complete collapse)
- Should score well on CT-level metrics (gene corr, classification)
- Perfect floor baseline — any method must beat this

## Priority order

1. **Cell-type mean** — trivial, implement first as sanity check (~1 hour)
2. **scVI** — highest reviewer demand, well-documented API (~1 day)
3. **Harmony + kNN** — reviewer-requested, quick to implement (~half day)
4. **scGen** — natural comparison to our perturbation framing (~half day)
5. **scANVI** — minimal extra work once scVI is set up (~2 hours)

## Expected results narrative

We expect:
- **Cell-type mean:** Perfect CT classification, zero variance retention — shows CT-level metrics are necessary but not sufficient
- **scVI transform_batch:** Good CT-level metrics, moderate variance retention — strong baseline but treats modality as nuisance batch (loses biological signal)
- **Harmony + kNN:** Moderate CT metrics, some variance retention from kNN — shows embedding alignment ≠ translation
- **scGen:** Good CT metrics, decent variance retention (preserves input variation + constant shift) — but shift is cell-type-uniform, misses gene-specific patterns
- **Our cVAE:** Best combination of CT-level accuracy AND gene-specific, non-uniform translation — learned per-gene adjustments (ribosomal up, lncRNA down) that other methods miss

## Environment setup

```bash
# Create a dedicated baseline env (to avoid dependency conflicts with impute2)
conda create -n baselines python=3.12 -y
conda activate baselines
pip install scvi-tools scanpy anndata harmonypy scgen
pip install torch --index-url https://download.pytorch.org/whl/cu118

# OR install into impute2 (simpler but riskier)
conda activate impute2
pip install scvi-tools scgen harmonypy
```

## Script organization

```
analysis/baselines/
  __init__.py
  data_loader.py      # Shared data loading (reuse from baseline_linear_regression.py)
  scvi_baseline.py     # scVI + scANVI
  scgen_baseline.py    # scGen
  harmony_baseline.py  # Harmony + kNN
  ctmean_baseline.py   # Cell-type mean
  run_all.py           # Orchestrator: run all baselines, collect results
  evaluate.py          # Shared evaluation (calls analysis/eval/ metrics)
```
