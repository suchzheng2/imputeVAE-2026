---
name: UMAP refactor for conditioned decoder
description: Plan to reorganize UMAP plots into train/valid subdirs with separate latent, decoded, cross-decoded, and cell_vs_cross plots. Fix batch ID bug where cell samples show nuclei batch IDs.
type: project
---

## UMAP Refactor Plan (2026-03-26)

**Why:** The conditioned decoder introduces cross-decoding (nuclei encoded + cell embedding). Current UMAPs only show encoded/decoded with cell+nuclei mixed. Need separate plots to evaluate cross-decoded quality. Also batch IDs are wrong for cell samples in mixed plots.

**How to apply:**

### New folder structure
```
umap/
  train/
    latent/          # cell + nuclei encoded (3 panels: celltype, batch, modality)
    decoded_cell/    # cell decoded only (2 panels: celltype, batch)
    decoded_nuclei/  # nuclei decoded only (2 panels: celltype, batch)
    cross_decoded/   # nuclei cross-decoded only (2 panels: celltype, batch)
    cell_vs_cross/   # decoded cell + cross-decoded nuclei (3 panels: celltype, batch, modality="cell" vs "cross-decoded nuclei")
  valid/
    (same subfolders)
```

### Batch ID fix
- Current bug: `meta_batch_id_original` uses nuclei batch IDs for both cell and nuclei samples
- Fix: use `cell_meta_batch_id_original` for cell samples, `nuclei_meta_batch_id_original` for nuclei samples
- Batch ID format: `{n|c}{technology}{individual}` e.g. n3A=nuclei, c3A=cell

### Files to modify
- `imputeVAE/visualization.py` — new UMAP functions for cond decoder
- `imputeVAE/cond_training.py` — call new UMAP functions at eval epochs
