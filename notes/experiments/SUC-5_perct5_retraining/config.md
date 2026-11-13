# SUC-5: Configuration

## Gene List

- File: `data/gene_list_perct_5pct.csv`
- Source: `notes/gene_filtering_analysis/gene_list_perct_5pct.csv`
- 16,439 genes (vs Config B's 10,134)
- Filter: ≥5% detection in any cell type in either modality

## Models

All use: vae_js (MSE), internal_size=10,9 (1024,512), embed_dim=8, batch_size=128, lr=1e-4, beta=0.5, subset_ratio=0.2, protocol=v3

### Stage 1: Cell pretraining

- Run: `run/run_2026-04-02-19-27-35_3767601`
- Job: 3134209 (COMPLETED, 4h 2m)
- Script: `jobs/stage1/run_s1_cell_js_perct5.sh`
- Data: 10X_cells_v2_v3_AIBS/, batch=c3A
- 600 epochs, JSD anneal start=300 over 100 epochs

### Stage 2: Cond decoder (no xr)

- Run: `run/run_2026-04-03-03-44-06_3047531`
- Job: 3175005 (CANCELLED at epoch 249) → 3235987 (RESUMED from epoch 250)
- Script: `jobs/stage2/run_s2_cond_js_perct5.sh` (original), `jobs/stage2/resume_s2_cond_js_perct5.sh` (resume)
- Data: 10X_nuclei_v2_v3_AIBS/, batch=n3A (target=nuclei, reference=cell)
- Loss schedule:
  - cell_recons=1.0 (always on), cell_jsd=0.5 (always on)
  - nuclei_recons=1.0, anneal 0→100
  - nuclei_jsd=0.5, delayed 100, anneal 100→200
  - alignment=1.0, delayed 200, anneal 200→300
  - cross_recons=0.0 (off)

### Stage 2: Cond decoder + cross-recons

- Run: `run/run_2026-04-03-03-44-41_2957436`
- Job: 3175006 (PREEMPTED at epoch 249) → 3235988 (RESUMED from epoch 250)
- Script: `jobs/stage2/run_s2_cond_xr_js_perct5.sh` (original), `jobs/stage2/resume_s2_cond_xr_js_perct5.sh` (resume)
- Same as above except: cross_recons=1.0, delayed 100, anneal 100→200

## Comparison Models (Config B baseline)

- MSE cond (Config B): `run/run_2026-04-02-02-02-35_3345839`
- MSE cond+xr (Config B): `run/run_2026-04-02-10-51-32_3738876`
