# Plan: Modality-Conditioned Decoder (cVAE)

## Goal

Train a VAE where the decoder receives a **modality embedding** alongside the latent code `z`. During training, each modality reconstructs with its own embedding (cell data → cell embedding, nuclei data → nuclei embedding). At evaluation time, we can feed nuclei latent codes with the **cell embedding** to produce cell-like output from nuclei data — without needing cross-recons loss.

## Why This Should Work

The current shared decoder has no way to distinguish modalities — it sees only `z` and must produce output that satisfies both cell and nuclei reconstruction simultaneously, which degrades quality. By conditioning on modality, the decoder can learn modality-specific adjustments (e.g., library size scale, gene proportion differences) while sharing the bulk of its parameters. At eval time, swapping the embedding is a clean, principled way to convert between modalities.

---

## Architecture Changes

### New model class: `VAE_NB_Cond` (in new file `imputeVAE/models_cond.py`)

Extends `VAE_NB`. Changes only the decoder input — encoder is untouched.

```
Encoder (unchanged):
  x [B, G] → encoder → encoded [B, L] → mu, log_var [B, L] → z [B, L]

Decoder (modified):
  z [B, L] + modality_emb [B, E] → concat → [B, L+E] → decoder → rho [B, G]
  mu_nb = library_size * softmax(rho)
```

**Modality embedding:**
- `nn.Embedding(2, embed_dim)` where index 0 = cell, 1 = nuclei
- `embed_dim` = hyperparameter, default 8 (small — just enough to condition, not dominate)
- The embedding is concatenated to `z` before the first decoder layer

**Decoder architecture change:**
- First decoder layer input changes from `num_latent` (32) to `num_latent + embed_dim` (32 + 8 = 40)
- All other layers unchanged: `40 → 512 → 1024 → num_genes`

**Key methods:**

```python
class VAE_NB_Cond(VAE_NB):
    def __init__(self, input_size, num_internal, num_latent, embed_dim=8, use_zinb=False):
        super().__init__(input_size, num_internal, num_latent, use_zinb)
        self.embed_dim = embed_dim
        self.modality_emb = nn.Embedding(2, embed_dim)  # 0=cell, 1=nuclei

        # Rebuild decoder with expanded input: num_latent + embed_dim
        decoder_layers = []
        in_dim = num_latent + embed_dim  # <-- only change
        for h in reversed(num_internal):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.LeakyReLU(0.1))
            in_dim = h
        decoder_layers.append(nn.Linear(num_internal[0], input_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, modality_id):
        """
        x: [B, G] raw counts
        modality_id: int (0=cell, 1=nuclei), applied to entire batch
        """
        # Encode (same as VAE_NB)
        encoded = self.encoder(x)
        mu_latent = self.mu(encoded)
        log_var_latent = self.log_var(encoded)
        log_var_latent = torch.clamp(log_var_latent, min=-20, max=20)
        std = torch.exp(log_var_latent * 0.5).clamp(min=1e-6)
        z = mu_latent + torch.randn_like(std) * std

        # Condition decoder on modality
        emb = self.modality_emb(
            torch.full((x.size(0),), modality_id, dtype=torch.long, device=x.device)
        )  # [B, E]
        z_cond = torch.cat([z, emb], dim=1)  # [B, L+E]

        # Decode
        rho = self.decoder(z_cond)
        library_size = x.sum(dim=1, keepdim=True).clamp(min=1.0)
        mu_nb = library_size * F.softmax(rho, dim=1)

        theta = F.softplus(self.log_theta).clamp(min=1e-4, max=1e6)
        self._nb_params = {"theta": theta}

        return encoded, mu_nb, mu_latent, log_var_latent

    def decode_cross(self, x_source, source_modality_id, target_modality_id):
        """
        Encode with source modality, decode with target modality embedding.
        Used at eval time: encode nuclei (id=1), decode with cell embedding (id=0).
        Returns mu_nb using source library size (caller can rescale if needed).
        """
        with torch.no_grad():
            encoded = self.encoder(x_source)
            mu_latent = self.mu(encoded)
            # Use mu directly (no sampling) for deterministic eval
            emb = self.modality_emb(
                torch.full((x_source.size(0),), target_modality_id,
                           dtype=torch.long, device=x_source.device)
            )
            z_cond = torch.cat([mu_latent, emb], dim=1)
            rho = self.decoder(z_cond)
            library_size = x_source.sum(dim=1, keepdim=True).clamp(min=1.0)
            mu_nb = library_size * F.softmax(rho, dim=1)
        return mu_nb
```

### Weight initialization from pretrained model

The encoder, mu/log_var heads, log_theta, and all decoder layers **except the first** have the same shape as the pretrained stage 1 model. Loading strategy:

```python
pretrained = torch.load(stage1_model_path)
# Load everything that matches
model_dict = model.state_dict()
matched = {k: v for k, v in pretrained.items()
           if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(matched)
model.load_state_dict(model_dict)
```

This will load:
- encoder (all layers) ✓
- mu, log_var heads ✓
- log_theta ✓
- decoder layers 2+ (Linear(512→1024), Linear(1024→G)) ✓
- decoder layer 0 (Linear(32→512)) ✗ — shape mismatch (now 40→512)

For decoder layer 0, we initialize by copying the pretrained weights into the first `num_latent` columns and zero-initializing the `embed_dim` columns:

```python
# decoder.0.weight: pretrained [512, 32] → new [512, 40]
old_w = pretrained["decoder.0.weight"]  # [512, 32]
old_b = pretrained["decoder.0.bias"]    # [512]
new_w = model.decoder[0].weight.data    # [512, 40]
new_w[:, :num_latent] = old_w           # copy latent portion
new_w[:, num_latent:] = 0.0             # zero-init embedding portion
model.decoder[0].bias.data = old_b      # bias unchanged
```

This ensures the model starts from the pretrained decoder behavior — the embedding columns are zero so the initial output is identical to the pretrained model regardless of modality ID.

---

## Training

### New training file: `imputeVAE/cond_training.py`

Separate from `staged_training.py`. Reuses the same data loading, dataloader creation, and validation infrastructure.

### Training loop overview

```
for epoch in range(num_epochs):
    for celltype in common_celltypes:
        cell_batch = next(cell_loader[celltype])
        nuclei_batch = next(nuclei_loader[celltype])

        # Forward pass — each modality with its own embedding
        cell_enc, cell_dec, cell_mu, cell_logvar = model(cell_batch, modality_id=0)
        nuc_enc, nuc_dec, nuc_mu, nuc_logvar = model(nuclei_batch, modality_id=1)

        # Losses (same as current stage 3, minus cross_recons)
        cell_recons_loss = NB_loss(cell_dec, cell_batch)
        cell_jsd = JSD(cell_mu, cell_logvar)
        nuc_recons_loss = NB_loss(nuc_dec, nuclei_batch)
        nuc_jsd = JSD(nuc_mu, nuc_logvar)
        alignment_jsd = JSD_two(cell_mu, cell_logvar, nuc_mu, nuc_logvar)

        loss = (cell_recons_w * cell_recons_loss
              + cell_jsd_w * cell_jsd
              + nuc_recons_w * nuc_recons_loss
              + nuc_jsd_w * nuc_jsd
              + align_w * alignment_jsd)

        loss.backward()
        optimizer.step()
```

### Loss components (5 terms, no cross-recons)

| Loss | Weight | Anneal | Purpose |
|------|--------|--------|---------|
| cell_recons | 1.0 | no | Reconstruct cell data with cell embedding |
| nuclei_recons | 1.0 | no | Reconstruct nuclei data with nuclei embedding |
| cell_jsd | 0.5 | no | Cell latent regularization |
| nuclei_jsd | 0.5 | no | Nuclei latent regularization |
| alignment | 3.0 | 0→3.0 over 100 epochs | Align latent distributions |

**No cross-recons needed.** The modality embedding handles the conversion — at eval time we just swap the embedding.

### Data loading

Same as current stage 3:
- Target data: nuclei (from `args.data_path`)
- Reference data: cell (auto-detected)
- Common cell types: intersection of both
- Paired by cell type, batch_size=128, subset_ratio=0.2

### Hyperparameters to expose

| Arg | Default | Description |
|-----|---------|-------------|
| `--embed_dim` | 8 | Modality embedding dimension |
| Everything else | same as current stage 3 | Reuse existing args |

---

## Evaluation

We now have **5 data sources** in output space (plus 2 in latent space):

| Label | What it is | How it's produced |
|-------|-----------|-------------------|
| `raw_cell` | Raw cell counts | Data (no model) |
| `raw_nuc` | Raw nuclei counts | Data (no model) |
| `dec_cell` | Cell-decoded cell | cell input → encoder → z → decoder(z, cell_emb) |
| `dec_nuc` | Nuclei-decoded nuclei | nuc input → encoder → z → decoder(z, nuc_emb) |
| `cross_dec` | Cross-decoded nuclei | nuc input → encoder → z → decoder(z, **cell_emb**) |

And in latent space:
| `enc_cell` | Cell encoded | cell input → encoder → mu |
| `enc_nuc` | Nuclei encoded | nuc input → encoder → mu |

### UMAPs

**Latent space (same as current):**
- `enc_cell + enc_nuc` together, colored by cell type — shows alignment quality

**Output space — reconstruction quality (per modality):**
- `dec_cell` alone, colored by cell type — cell reconstruction quality
- `dec_nuc` alone, colored by cell type — nuclei reconstruction quality

**Output space — both modalities together:**
- `dec_cell + dec_nuc` together, colored by modality+cell_type — shows if decoder separates modalities (we expect separation here since embeddings differ)

**Output space — cross-decode (the key plots):**
- `cross_dec + dec_cell` together, colored by modality — does cross-decoded nuclei land on top of cell-decoded cells? This is the main success metric
- `cross_dec + raw_cell` together, colored by modality — does cross-decoded nuclei resemble real cells?
- `cross_dec + dec_nuc` together, colored by modality — how far did the cross-decode shift from normal nuclei decode?

**Summary of UMAP plots per eval epoch:**

| Plot | Sources | Color by | Question answered |
|------|---------|----------|-------------------|
| `umap_encoded.png` | enc_cell + enc_nuc | cell_type + modality | Latent alignment? |
| `umap_dec_cell.png` | dec_cell | cell_type | Cell recon quality? |
| `umap_dec_nuc.png` | dec_nuc | cell_type | Nuclei recon quality? |
| `umap_dec_both.png` | dec_cell + dec_nuc | cell_type + modality | Modality separation in output? |
| `umap_cross_vs_dec_cell.png` | cross_dec + dec_cell | modality | Cross-dec ≈ cell-dec? |
| `umap_cross_vs_raw_cell.png` | cross_dec + raw_cell | modality | Cross-dec ≈ real cells? |
| `umap_cross_vs_dec_nuc.png` | cross_dec + dec_nuc | modality | How much did cross-dec shift? |

### Correlation Matrix

Expand from 4×4 to **5×5**: raw_cell, raw_nuc, dec_cell, dec_nuc, **cross_dec**.

```
              raw_cell  raw_nuc  dec_cell  dec_nuc  cross_dec
raw_cell         1.0      —        —         —        —
raw_nuc          —        1.0      —         —        —
dec_cell         —        —        1.0       —        —
dec_nuc          —        —        —         1.0      —
cross_dec        —        —        —         —        1.0
```

Key cells to watch:
- `cross_dec ↔ raw_cell` — does cross-decoded output match real cell distribution?
- `cross_dec ↔ dec_cell` — does cross-decoded output match cell-decoded output?
- `cross_dec ↔ raw_nuc` — has it shifted away from nuclei?
- `dec_cell ↔ raw_cell` — cell reconstruction fidelity (should be high)
- `dec_nuc ↔ raw_nuc` — nuclei reconstruction fidelity (should be high)

### Shift Scores

Compute shift scores for **3 decoded outputs** (not just 2):
- `dec_cell` shift — cell reconstruction fidelity
- `dec_nuc` shift — nuclei reconstruction fidelity
- `cross_dec` shift — **the key metric**: should be near zero or positive (cell-like)

### Gene Correlation

Per-gene correlation for all decoded outputs vs both raw references:
- `dec_cell` vs `raw_cell` (cell→cell, should be high)
- `dec_nuc` vs `raw_nuc` (nuc→nuc, should be high)
- `cross_dec` vs `raw_cell` (cross→cell, the conversion metric)
- `cross_dec` vs `raw_nuc` (cross→nuc, how much nuclei signal remains)

### Decoder Scale

Mean decoded output per source:
- `dec_cell` mean — should match cell library size scale (~1.9)
- `dec_nuc` mean — should match nuclei library size scale (~0.66)
- `cross_dec` mean — interesting: will it shift toward cell scale or stay at nuclei scale?
  (Note: library size comes from input, so `cross_dec` uses nuclei library size. The scale will reflect nuclei library size × cell-like proportions. This is expected and correct.)

### Validation

Validation loop computes all the above on held-out data at each eval epoch:
- Validation loss (5 components)
- Validation cross-decode metrics (correlation, shift, gene corr)
- Train vs valid comparison to check overfitting

### What we expect to see

- **Reconstruction quality** should be excellent (better than current stage 3) because each modality has its own conditioning signal — decoder doesn't compromise
- **Cross-decoded nuclei** should resemble cells if the latent alignment worked and the decoder learned meaningful modality-specific adjustments via the embedding
- The degree of cell-likeness depends on how much modality information the embedding captures vs how much leaks into `z`
- If alignment is good but cross-decode is bad → embedding is not learning modality differences (try larger embed_dim)
- If alignment is good and cross-decode is good → success

---

## File Structure

### New files to create

| File | Purpose |
|------|---------|
| `imputeVAE/models_cond.py` | `VAE_NB_Cond` model class |
| `imputeVAE/cond_training.py` | Training loop for conditioned model |
| `jobs/stage3/train_cond_probe.sh` | SLURM script (probe: 100 epochs) |
| `jobs/stage3/train_cond_cell2both.sh` | SLURM script (full: 600 epochs, after probe) |

### Files to modify

| File | Change |
|------|--------|
| `train.py` | Add `--fine_tune_cond_stage3` flag + `--embed_dim` arg; route to `cond_training` |
| `imputeVAE/__init__.py` | Export new model and training function |

### Files NOT modified

- `models.py` — untouched, `VAE_NB` stays as-is
- `staged_training.py` — untouched, existing stage 3 preserved

---

## Start Point

Same stage 1 cell model as all previous stage 3 runs:
- `run/run_2026-03-12-17-29-06_928562/models/model.pt`

Partial weight loading as described above (all weights except decoder layer 0, which gets expanded).

---

## SLURM Scripts

### Probe (100 epochs, subset_ratio=0.1)

```bash
#!/bin/bash
#SBATCH --job-name=cond-probe
#SBATCH --out="jobs/logs/slurm-%j_cond_probe.out"
#SBATCH --partition=gpu,pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --mem=200G
#SBATCH --mail-type=ALL

module load miniconda
source activate impute2

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0

python train.py \
   -dp "10X_nuclei_v2_v3_AIBS/" \
   --model "vae_nb" \
   -b "n3A" \
   --protocol "v3" \
   --use_raw_data True \
   --raw_data_normalization "raw" \
   --fine_tune_cond_stage3 True \
   --stage1_model_run "run/run_2026-03-12-17-29-06_928562/models/model.pt" \
   --internal_size "10,9" \
   --latent_size 5 \
   --embed_dim 8 \
   --num_epochs 100 \
   --cell_recons_weight 1.0 \
   --cell_jsd_weight 0.5 \
   --nuclei_recons_weight 1.0 \
   --nuclei_jsd_weight 0.5 \
   --alignment_weight 3.0 \
   --anneal_alignment True \
   --anneal_alignment_start_epoch 0 \
   --anneal_alignment_epoch 50 \
   --batch_size 128 \
   --learning_rate 1e-4 \
   --subset True \
   --subset_ratio 0.1 \
   --exclude_mitocarta True \
   --stage3_eval_every 25 \
   --stage3_save_epoch_dump False \
   --debug True
```

### Full run (600 epochs, subset_ratio=0.2) — after probe validates

```bash
#!/bin/bash
#SBATCH --job-name=cond-c2b
#SBATCH --out="jobs/logs/slurm-%j_cond_cell2both.out"
#SBATCH --partition=gpu,pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --mem=500G
#SBATCH --mail-type=ALL

module load miniconda
source activate impute2

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0

python train.py \
   -dp "10X_nuclei_v2_v3_AIBS/" \
   --model "vae_nb" \
   -b "n3A" \
   --protocol "v3" \
   --use_raw_data True \
   --raw_data_normalization "raw" \
   --fine_tune_cond_stage3 True \
   --stage1_model_run "run/run_2026-03-12-17-29-06_928562/models/model.pt" \
   --internal_size "10,9" \
   --latent_size 5 \
   --embed_dim 8 \
   --num_epochs 600 \
   --cell_recons_weight 1.0 \
   --cell_jsd_weight 0.5 \
   --nuclei_recons_weight 1.0 \
   --nuclei_jsd_weight 0.5 \
   --alignment_weight 3.0 \
   --anneal_alignment True \
   --anneal_alignment_start_epoch 0 \
   --anneal_alignment_epoch 100 \
   --batch_size 128 \
   --learning_rate 1e-4 \
   --subset True \
   --subset_ratio 0.2 \
   --exclude_mitocarta True \
   --stage3_eval_every 50 \
   --stage3_save_epoch_dump True \
   --stage3_dump_every 50 \
   --debug True
```

Note: `--model "vae_nb"` is kept for arg parsing compatibility — the `--fine_tune_cond_stage3` flag tells `train.py` to create a `VAE_NB_Cond` instead.

---

## Implementation Steps

1. **Create `imputeVAE/models_cond.py`**
   - `VAE_NB_Cond` class with modality embedding + modified decoder input
   - `forward(x, modality_id)` — standard forward with embedding
   - `decode_cross(x_source, source_id, target_id)` — cross-modality decode for eval
   - `load_pretrained_partial(path)` — weight loading with first-layer expansion + zero-init
   - `loss_function()` — inherited from VAE_NB, unchanged

2. **Create `imputeVAE/cond_training.py`**
   - `train_cond_stage3(adata, cell_types, save_path, args)` — main training loop
   - Import data loading helpers from `staged_training.py`
   - Training loop: forward both modalities with their IDs, compute 5 losses
   - Inline eval at eval epochs:
     - Standard: decoder scale, shift scores, gene corr, UMAPs (reuse existing functions)
     - Cross-decode: compute `decode_cross()`, then 5×5 correlation, cross-decode shift, cross-decode gene corr, 7 UMAP plots
   - Validation loop: same metrics on held-out data
   - Checkpoint saving, loss logging, weight annealing — same infrastructure as `staged_training.py`

3. **Modify `train.py`**
   - Add `--fine_tune_cond_stage3` (str2bool, default False)
   - Add `--embed_dim` (int, default 8)
   - Add routing block: `if args.fine_tune_cond_stage3: ConditionalVAE.train_cond_stage3(...); exit()`

4. **Update `imputeVAE/__init__.py`**
   - Export `train_cond_stage3` from `cond_training`

5. **Create SLURM scripts**
   - `jobs/stage3/train_cond_probe.sh` — 100 epochs, subset_ratio=0.1, eval every 25
   - `jobs/stage3/train_cond_cell2both.sh` — 600 epochs, subset_ratio=0.2 (after probe validates)

6. **Run probe** — 100 epochs, subset_ratio=0.1
   - Verify reconstruction quality matches/exceeds current stage 3 baseline
   - Check cross-decoded output: does it shift toward cells?
   - Compare alignment convergence
   - Review UMAPs: do per-modality decoded UMAPs retain cell type clusters?

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Embedding too small (8 dims) → decoder can't distinguish modalities | Try embed_dim=16 or 32 in follow-up probes |
| Embedding too large → decoder ignores latent code, relies entirely on embedding | Keep embed_dim small (8); zero-init ensures gradual learning |
| Latent space encodes modality info despite alignment → cross-decode doesn't work | Alignment loss pushes z distributions together; can increase alignment weight |
| Decoder learns to ignore embedding → no benefit over current approach | Monitor embedding gradient norms; if near-zero, increase embed_dim or add embedding regularization |

## Comparison to Current Approach

| Aspect | Current (cross-recons) | Conditioned decoder |
|--------|----------------------|-------------------|
| Decoder conflict | Yes — single decoder serves both modalities | No — embedding resolves ambiguity |
| Cross-modality signal | Explicit loss term (NB cross-recons) | Implicit via embedding swap at eval |
| Reconstruction quality | Degraded (washed-out UMAPs) | Expected to be better |
| Conversion mechanism | Force decoder output to match other modality | Swap conditioning signal |
| Extra parameters | 0 | embed_dim × 2 + first decoder layer expansion (~320 params for embed_dim=8) |
| Training complexity | 6 loss terms | 5 loss terms (simpler) |
