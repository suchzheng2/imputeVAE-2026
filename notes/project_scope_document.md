# imputeVAE — Full Project Scope Document

> Use this as a **Project Knowledge** document in the Claude app to give a new session complete context on the imputeVAE project.

---

## 1. What This Project Does

**imputeVAE** is a VAE-based framework for **translating single-nucleus RNA-seq (snRNA-seq) profiles into cell-like expression profiles**. The biological motivation: snRNA-seq is cheaper and works on frozen tissue, but systematically differs from scRNA-seq (whole-cell) — nuclei miss cytoplasmic RNA, have different library sizes, and show biased gene detection. This project trains a shared latent space across both modalities and learns to decode nucleus latents into cell-like expression.

**Lab:** Mark Gerstein's group at Yale. **Platform:** Yale HPC (SLURM). **Species:** Mouse brain (Allen Brain Cell Atlas data, 10X Chromium v3 chemistry).

---

## 2. Architecture & Training Pipeline

### Models

Two VAE families, both with symmetric encoder-decoder (LeakyReLU activations):

| Model | Loss | Data format | Key trait |
|-------|------|-------------|-----------|
| **VAE_JSD** (MSE) | MSE reconstruction + JSD divergence | log1p-normalized | Each gene independent, no bottleneck. **Currently preferred.** |
| **VAE_NB** (NB) | Negative binomial likelihood + JSD | Raw counts | Softmax output → zero-sum bottleneck where dominant genes steal probability mass. Deprecated. |

Both have conditioned-decoder variants (`VAE_JSD_Cond`, `VAE_NB_Cond`) that add a learned modality embedding (dim=8) to the decoder input, enabling modality-specific decoding from a shared latent space.

**Standard architecture:** Input [G genes] → [1024] → [512] → [32 latent] → [512] → [1024] → [G genes]

### Three-Stage Training

**Stage 1 — Single-modality pretraining (cell only):**
- Train VAE on cell data only (600 epochs)
- Reconstruction loss + JSD divergence (annealed: 0 for 300 epochs, ramp over 100, full for 200)
- Produces a cell-specific encoder/decoder

**Stage 2 — Joint training (cell + nuclei):**
- Load Stage 1 model, introduce nuclei data
- **Cascading loss annealing** — losses turned on sequentially to avoid competing gradients:
  - Epoch 0: cell_recons + cell_jsd (always on)
  - Epoch 0–100: nuclei_recons ramps from 0 → 1.0
  - Epoch 100–200: nuclei_jsd ramps from 0 → 0.5
  - Epoch 200–300: alignment JSD ramps from 0 → 1.0
  - Epoch 100–200 (optional): cross_recons ramps from 0 → 1.0
- **Alignment loss:** JSD between cell and nuclei latent distributions (pushes them into shared space)
- **Cross-reconstruction loss (optional):** Encode nuclei → decode with cell embedding → MSE against real cell data

**Evaluation:** UMAPs, shift scores, gene correlation, classification accuracy, DE preservation, scIB metrics, variance ratio tests.

### Conditioned Decoder

The key architectural innovation. Problem: a shared decoder serving cell_recons + nuclei_recons + cross_recons + alignment gets "pulled in too many directions" and produces degraded output. Solution: add `Embedding(2, 8)` (cell=0, nuclei=1) concatenated to z before decoding. The decoder can specialize per modality while sharing most parameters.

**Cross-decode at inference:** Encode nucleus → z → decode with cell embedding → cell-like output.

---

## 3. Key Results & Current State

### Best Models (Config B, 10,134 genes — older gene set)

| Variant | Classification | DE Preservation | Coexpression | Variance Ratio |
|---------|---------------|----------------|-------------|---------------|
| MSE cond+xr | **95.8%** (vs 85.1% raw nuclei) | **+5.1%** | 0.891 | 3% |
| MSE cond (no xr) | 93.7% | +4.8% | **0.916** | 11.5% |
| MSE shared | 85.1% | +0% | 0.818 | — |

### The Variance Collapse Problem (Main Challenge)

Models achieve excellent **cell-type-level** metrics (classification, DE, mean correlation) but collapse to cell-type means:
- **Variance ratio:** Imputed cells have only 3–11% of real cell within-cell-type variance
- **z ablation:** Replacing z with cell-type mean z changes output by <0.1% (decoder ignores per-cell info)
- **Within-CT Spearman:** 0.035–0.112 (near zero per-cell correlation)

**Root cause:** MSE loss on small batches converges to cell-type means; no paired cell-nucleus data for per-cell supervision; 32-dim latent uses most capacity for cell-type identity.

### Gene Filtering Evolution

| Filter | Genes | Issue |
|--------|-------|-------|
| det>1% (~24.7K) | Too many noise genes | Original, deprecated |
| Config B (det>5%, 10,134) | Missing 29/59 markers | Non-neuronal DE near zero |
| **PerCT_5pct (5% per CT, 16,439)** | Recovers 58/59 markers | **Current standard** |

### MSE vs NB Verdict

MSE (log1p) dominates NB (raw counts) on every metric. NB's softmax bottleneck suppresses low-expression genes. All new work uses MSE.

---

## 4. Active Work (as of 2026-04-06)

**Running jobs:**
- Job 3235987: PerCT_5pct Stage 2 cond (no xr), epoch ~382/600
- Job 3235988: PerCT_5pct Stage 2 cond+xr, epoch ~349/600

**Stage 1 (completed):** `run/run_2026-04-02-19-27-35_3767601` — PerCT_5pct, 16,439 genes

**Next steps after jobs complete:**
1. Run hallucination tests on PerCT_5pct models (variance ratio, z ablation)
2. Phase 1: Larger latent (64/128 dim) — just new job scripts
3. Phase 2: Variance-preserving regularization — code changes in cond_training.py
4. Phase 3: Residual reconstruction weighting
5. Phase 4: Read subsampling for paired data (most principled, most work)

---

## 5. Repository Structure

```
imputeVAE/
├── train.py                       # Entry point — arg parsing, dispatches to training loops
├── imputeVAE/                     # Core package
│   ├── models.py                  # VAE, VAE_JSD, VAE_NB base classes
│   ├── models_cond.py             # VAE_JSD_Cond, VAE_NB_Cond (modality embedding)
│   ├── data.py                    # Data loading, gene filtering, train/valid/test split
│   ├── training.py                # Stage 1 training loop
│   ├── staged_training.py         # Stage 2 joint training (shared decoder)
│   ├── cond_training.py           # Stage 2 joint training (conditioned decoder)
│   ├── divergences.py             # JSD (Monte Carlo), Wasserstein distance
│   ├── stage3_eval.py             # Inline evaluation during training
│   ├── testing.py                 # Post-training evaluation
│   ├── visualization.py           # UMAP plotting
│   ├── plot_loss.py               # Loss curve plotting
│   └── utils.py                   # GPU setup, helpers
├── analysis/                      # ~31 evaluation/analysis scripts
│   ├── eval_utils.py              # Shared eval utilities
│   ├── comprehensive_eval.py      # Large comparison runs
│   ├── hallucination_test.py      # Variance collapse analysis
│   ├── eval_scib.py               # scIB metrics
│   └── ...
├── jobs/                          # SLURM scripts
│   ├── stage1/                    # Stage 1 training
│   ├── stage2/                    # Stage 2 joint training
│   ├── probe/                     # Quick weight exploration (100 epochs)
│   ├── eval/                      # Post-training evaluation
│   └── logs/                      # SLURM stdout/stderr
├── data/                          # Gene lists, cell type lists
│   ├── gene_list_perct_5pct.csv   # Current standard (16,439 genes)
│   ├── training_gene_list_v3.csv  # Config B (10,134 genes, deprecated)
│   ├── common_genes.csv           # Shared cell/nuclei genes (25,802)
│   └── Mouse.MitoCarta.csv        # Mitochondrial exclusion list
├── run/                           # Training outputs (auto-generated)
│   └── run_YYYY-MM-DD-HH-MM-SS_{id}/
│       ├── arguments.json         # Hyperparameters
│       ├── models/model.pt        # Checkpoint
│       ├── epoch_dump/            # Per-epoch data
│       └── umap/, loss_curves/, evaluation/
├── notes/                         # Project documentation
│   ├── training_runs.md           # Master run inventory
│   ├── design_decisions.md        # Architecture rationale
│   ├── next_steps.md              # Variance improvement plan
│   └── ...
└── cache/                         # Cached dataloaders
```

---

## 6. Key CLI Arguments (train.py)

**Data:**
- `--data_path`: e.g., "10X_cells_v2_v3_AIBS/"
- `--protocol v3`: filter to 10X v3 chemistry
- `--batch c3A` / `n3A`: cell/nuclei batch
- `--gene_list data/gene_list_perct_5pct.csv`: gene filtering
- `--use_raw_data True --raw_data_normalization log1p`: for MSE models

**Model:**
- `--model vae_js`: MSE model (preferred)
- `--internal_size 10,9`: hidden layers [1024, 512]
- `--latent_size 5`: latent dim = 2^5 = 32
- `--embed_dim 8`: modality embedding dimension

**Training:**
- `--num_epochs 600`, `--batch_size 128`, `--learning_rate 1e-4`
- `--beta 0.5`: JSD weight
- `--kl_anneal True --kl_anneal_start_epoch 300 --kl_anneal_epoch 100`

**Stage 2:**
- `--fine_tune_cond_stage3 True`: conditioned decoder
- `--stage1_model_run <path>`: pretrained model
- `--cell_recons_weight 1.0 --nuclei_recons_weight 1.0 --alignment_weight 1.0`
- `--cross_recons_weight 1.0`: enable cross-reconstruction
- `--anneal_nuclei_recons_weight True --nuclei_recons_weight_start_epoch 0 --nuclei_recons_weight_anneal_epoch 100`

**Continue training:**
- `--continue_training True --continue_training_run_path <path> --continue_training_epoch N`

---

## 7. Key Design Decisions (Chronological)

1. **V3-only data** (2026-03-31): Dropped v2 chemistry due to strong batch effect (3–5x library size difference).

2. **MSE over NB** (2026-03-31): NB's softmax bottleneck suppresses low-expression genes. MSE on log1p gives each gene independent reconstruction.

3. **Cascading loss annealing** (2026-03-31): Losses introduced sequentially (cell_recons → nuclei_recons → JSD → alignment → cross_recons) to avoid competing gradients early in training.

4. **Conditioned decoder** (2026-03-24): Modality embedding solves decoder degradation from shared decoder serving competing objectives.

5. **Cross-recons is the key ingredient** (2026-03-20): Probe experiments showed cross-recons provides 3x better shift scores than any weight rebalancing alone. It's the only mechanism that directly supervises "decoded nuclei should look like cells."

6. **PerCT_5pct gene filtering** (2026-04-02): Config B (global 5%) missed 29/59 cell-type markers. Per-cell-type 5% detection recovers 58/59 markers (16,439 genes).

7. **Deprioritizing cross-recons** (2026-04-06): Cross-recons compresses within-cell-type variance from 11% to 3%. Focus shifted to per-cell variation preservation.

---

## 8. Experimental History Summary

### Weight Exploration Probes (A–E, 100 epochs each)
- Cross-recons is the critical ingredient (Probes D, E dramatically outperformed A, B, C)
- Weight rebalancing alone insufficient

### F-Series Probes (F1–F6, cross-recons weight sweep)
- Higher cross_recons improves cell-likeness monotonically
- F6 (xr=2.0, nuc_recons=0.5) best shift but worst nuclei fidelity
- F3 (xr=2.0, nuc_recons=1.0) best balanced

### MSE vs NB (full 600-epoch comparison)
- MSE dominates on every metric
- NB softmax bottleneck confirmed as limitation

### Conditioned Decoder Experiments
- Cond without xr ≈ shared decoder (no improvement)
- Cond + xr = best classification (95.8%) and DE (+5.1%)
- But cond+xr collapses variance to 3%

### Hallucination Tests
- All models collapse to cell-type means
- z ablation correlation 0.998–0.999 (decoder ignores per-cell z variation)
- This is the main open problem

---

## 9. Environment & Operations

```bash
# Activate environment
conda activate impute2

# Submit a job
sbatch jobs/stage2/run_s2_cond_js_perct5.sh

# Check job status
sacct -j JOBID --format=JobID,JobName%20,State,Elapsed -n

# Check training progress
grep "Epoch.*completed" jobs/logs/slurm-JOBID_*.out | tail -5

# Check errors
tail -20 jobs/logs/slurm-JOBID_*.err

# Important: delete __pycache__ before submitting jobs (stale bytecode has caused failures)
find . -type d -name __pycache__ -exec rm -rf {} +
```

**Partitions:** `pi_gerstein` (reliable), `scavenge_gpu` (faster but preemptible — use `--continue_training` to resume)

**Data locations (outside repo):**
- `/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/scvi/10X_cells_v2_v3_AIBS/data.h5ad`
- `/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/scvi/10X_nuclei_v2_v3_AIBS/data.h5ad`

---

## 10. Run Inventory (Key Models Only)

### Stage 1 Models

| Run | Model | Genes | Notes |
|-----|-------|-------|-------|
| `run_2026-03-12-17-29-06_928562` | vae_nb cell | ~24.7K | Original NB |
| `run_2026-04-01-09-29-26_3250320` | vae_js cell | 10,134 | Config B MSE |
| `run_2026-04-02-19-27-35_3767601` | vae_js cell | 16,439 | **PerCT_5pct MSE (current)** |

### Stage 2 Models (Config B)

| Run | Variant | Classification | DE | Var Ratio |
|-----|---------|---------------|-----|-----------|
| `run_2026-04-01-14-01-16_1946018` | MSE shared | 85.1% | +0% | — |
| `run_2026-04-02-02-02-35_3345839` | MSE cond | 93.7% | +4.8% | 11.5% |
| `run_2026-04-02-10-51-32_3738876` | MSE cond+xr | **95.8%** | **+5.1%** | 3% |

### Stage 2 Models (PerCT_5pct, in progress)

| Run | Variant | Status |
|-----|---------|--------|
| `run_2026-04-03-03-44-06_3047531` | MSE cond | Running (job 3235987) |
| `run_2026-04-03-03-44-41_2957436` | MSE cond+xr | Running (job 3235988) |

---

## 11. Open Questions & Future Directions

1. **Per-cell variation:** How to prevent collapse to cell-type means? Promising approaches: larger latent, variance regularization, read subsampling for pseudo-paired data.

2. **Gene filtering validation:** Does PerCT_5pct (16,439 genes) actually improve non-neuronal cell-type DE as expected?

3. **Cross-recons trade-off:** Is it possible to get cross-recons benefits (high classification, DE) without variance collapse?

4. **Scalability:** Current architecture handles 16K genes. Would scaling to full transcriptome (~25K) require different architecture?

5. **Biological validation:** How well do imputed profiles recover known cell-nucleus differences (cytoplasmic RNA, immediate early genes)?
