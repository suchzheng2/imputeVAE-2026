# Next Session Notes — Resume Here

## IMMEDIATE TODO: Compute intersection gene set and rerun training

### Context
Stage 1 jobs (3019128 MSE, 3019129 NB) completed successfully (600 epochs each) BUT used only cell-side detection rate filtering (11,714 genes). Stage 2 jobs (3056550, 3056551) failed because nuclei data produces 9,169 genes after filtering → shape mismatch when loading stage 1 checkpoint.

**User's decision:** The correct approach is:
1. Filter cell genes by detection rate ≥ 5% → cell gene set
2. Filter nuclei genes by detection rate ≥ 5% → nuclei gene set
3. **Intersect** the two → common training gene set
4. Exclude MitoCarta from the intersection
5. Use this gene list for BOTH stage 1 and stage 2

### Steps to execute

1. **Generate the intersection gene list** — script below, save to `data/training_gene_list_v3.csv`:
   ```python
   # Raw data path: /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/{dataset}/data.h5ad
   # Cell: 10X_cells_v2_v3_AIBS, filter meta_batch_id contains 'c3'
   # Nuclei: 10X_nuclei_v2_v3_AIBS, filter meta_batch_id contains 'n3'
   # Both: subset to common_genes.csv first, then compute det rate per gene
   # Intersect genes passing 5% in both → exclude MitoCarta → sort → save
   ```

2. **Cancel any running/pending jobs:**
   ```bash
   scancel 3056550 3056551  # already failed, but cleanup
   ```

3. **Update stage 1 scripts** to use `--gene_list data/training_gene_list_v3.csv`
   - `jobs/run_s1_cell_vae_js_v3.sh`
   - `jobs/run_s1_cell_vae_nb_v3.sh`
   - The `--gene_list` flag already exists in code (added to `train.py` and `data.py`)
   - When `--gene_list` is provided, it skips detection_rate and mitocarta filters

4. **Submit stage 1 jobs**, get new job IDs

5. **Update stage 2 scripts** with:
   - New stage 1 model paths (run folders use format `run/run_YYYY-MM-DD-HH-MM-SS_NODEID/`)
   - Add `--gene_list data/training_gene_list_v3.csv` so nuclei data uses same gene set
   - Submit with `--dependency=afterok:<stage1_job_id>`

### Code changes already made (in current codebase)

- **`imputeVAE/data.py`** line ~296: `--gene_list` support added. If provided, loads gene list and subsets `adata.var['gene_names']` to those genes, skipping detection rate + mitocarta filters.
- **`train.py`** line ~502: `--gene_list` argument added (type=str, default=None).
- **`imputeVAE/data.py`** line ~298: detection_rate_threshold bug fixed (was filtering cells axis=1, now filters genes axis=0, auto-converts percentage >1.0 to fraction).
- **`jobs/run_s2_joint_vae_js_v3.sh`**: Uses hardcoded STAGE1_MODEL path (no longer auto-discovers by job ID). Currently points to OLD stage 1 run — needs update after new stage 1.
- **`jobs/run_s2_joint_vae_nb_v3.sh`**: Same — hardcoded path, needs update.

### Completed stage 1 runs (OUTDATED — wrong gene set, cell-only filtering)
- MSE: `run/run_2026-04-01-01-16-21_3225695/` (11,714 genes, vae_js)
- NB: `run/run_2026-04-01-01-16-40_2163999/` (11,714 genes, vae_nb)
- These are NOT usable for stage 2 — gene count mismatch with nuclei

### Architecture and training schedule (unchanged)
- `--internal_size "10,9" --latent_size 5` → encoder [N, 1024, 512, 32], decoder [32, 512, 1024, N]
- Stage 1: 600 epochs (recons-only 0-300, JSD anneal 300-400, both 400-600)
- Stage 2: 600 epochs, cascading annealing (cell always on, nuclei_recons 0-100, nuclei_jsd 100-200, alignment 200-300)
- MSE: `--model vae_js --raw_data_normalization log1p --recons_loss_type 0`
- NB: `--model vae_nb --raw_data_normalization raw`
- Memory: 400G, partition: pi_gerstein,scavenge_gpu, time: 24h

### Key file locations
- Raw data: `/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/{dataset}/data.h5ad`
- Common genes: `data/common_genes.csv` (25,802 genes)
- MitoCarta: `data/Mouse.MitoCarta.csv` (Symbol column)
- Gene diff analysis results: `analysis/filtering_cutoffs_v3/`
- Job scripts: `jobs/run_s{1,2}_*_vae_{js,nb}_v3.sh`

### Bugs fixed this session
- detection_rate_threshold: axis=1→axis=0 (was filtering cells, now filters genes)
- `sp.issparse` NameError: replaced with `np.asarray(...).flatten()` (no scipy.sparse import)
- `--batch` flag required: added `c3A`/`n3A`
- `--use_raw_data True` required for vae_js
- OOM at 64G → 400G
- Stage 2 48h → 24h (scavenge_gpu limit)
- Run folder naming uses node ID not SLURM job ID → hardcoded paths in stage 2 scripts

### Key decisions
1. V3 only (drop v2 batch effect)
2. Config B filtering: det≥5%, mean>0.01, exclude MitoCarta
3. **NEW: Use intersection of cell+nuclei gene sets** (not just cell-side filtering)
4. MSE vs NB side-by-side comparison
5. Cell as reference modality in stage 2
