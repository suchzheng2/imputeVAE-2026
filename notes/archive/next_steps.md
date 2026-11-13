# Next Steps (2026-04-02)

## 1. Retrain with PerCT_5pct Gene Filtering

### Why
Config B (10,134 genes, 5% global detection) removes 29/59 brain cell-type markers including:
- Pvalb (Pvalb neurons), Sncg (Sncg), Aqp4 (astrocyte), Mog (oligodendrocyte)
- All OPC markers (Pdgfra, Cspg4, Olig1/2), all endothelial markers (Flt1, Cldn5, Pecam1)
- Most microglia markers (C1qa/b/c, Ctss, Tmem119)

This directly causes near-zero DE overlap for non-neuronal types (Astro=0.04, Endo=0.02, Oligo=0.06).

PerCT_5pct (16,439 genes, 5% detection in ANY cell type in EITHER modality) recovers 58/59 markers.
Only Aldh1l1 still missing. Gene list: `notes/gene_filtering_analysis/gene_list_perct_5pct.csv`

### Plan

**Stage 1: Pretrain on cell data**
- Copy existing stage 1 job script, change `--gene_list` to `data/gene_list_perct_5pct.csv`
- `--num_feature` will be 16439 instead of 10134
- Keep all other hyperparams the same
- Submit to scavenge_gpu, 600 epochs

**Stage 2a: MSE cond (no xr)**
- Joint training with conditioned decoder
- Use new stage 1 checkpoint as pretrained model
- Same cascading annealing schedule

**Stage 2b: MSE cond+xr**
- Same as 2a but with cross-recons loss enabled

### Files to create
1. Copy `data/gene_list_perct_5pct.csv` from `notes/gene_filtering_analysis/` to `data/`
2. New stage 1 job: `jobs/stage1/run_s1_cell_js_perct5.sh`
3. New stage 2 cond job: `jobs/stage2/run_s2_cond_js_perct5.sh`
4. New stage 2 cond+xr job: `jobs/stage2/run_s2_cond_xr_js_perct5.sh`

### Expected impact
- Non-neuronal DE should improve dramatically (markers now in gene set)
- Classification should improve (more discriminative genes)
- Total variance captured increases from 85% → 93%
- Model sees ~60% more genes — may need to adjust hidden layer size

---

## 2. Improving Per-Cell Variation

### The problem
Current models collapse to cell-type means:
- Variance ratio: 3% (cond+xr) to 11% (cond) of real cell variance
- z ablation: replacing z with CT mean changes output by <0.1%
- Within-CT Spearman: 0.035 (cond+xr) to 0.112 (cond)

The decoder learns to map z → cell-type identity → cell-type mean, ignoring per-cell variation in z.

### Why this happens
1. **MSE loss on cell-type means**: cross-recons loss computes MSE between cross-decoded output and target cell expression. But training batches are small relative to cell-type size, so the loss gradient pushes toward the batch mean, which converges to cell-type mean.
2. **No explicit per-cell supervision**: we don't have paired cell-nucleus data, so there's no "this specific nucleus should produce this specific cell profile."
3. **Information bottleneck**: the 32-dim latent z encodes 10K+ genes, and the decoder finds the easiest solution — use z for cell-type identity (which explains most variance) and modality embedding for the rest.

### Approaches to explore

#### A. Training objective changes (no architecture change)

**A1. Per-cell contrastive loss within cell types**
- Within a cell type, pull z's of similar cells closer, push dissimilar ones apart
- Forces z to encode within-CT variation, not just CT identity
- Implementation: sample pairs within CT, use cosine similarity loss
- Risk: may not transfer to cross-decode if decoder still ignores within-CT z

**A2. Adversarial loss on within-CT distributions**
- Train a discriminator: "can you tell imputed cells from real cells within this cell type?"
- Forces the generator (decoder) to produce realistic per-cell variation
- Similar to conditional GAN but applied within each CT
- Risk: training instability, mode collapse

**A3. Variance-preserving regularization**
- Add a loss term: penalize when Var(imputed | CT) << Var(real cell | CT)
- Simple: `L_var = max(0, target_ratio - Var(imputed)/Var(real))`
- Forces the decoder to maintain spread
- Risk: could produce noise rather than meaningful variation

**A4. Reconstruction weight on per-cell residuals**
- Decompose target into: cell-type mean + residual
- Weight the residual reconstruction more heavily
- Forces the model to care about per-cell deviations, not just the mean
- Implementation: `loss = MSE(output, target_mean) + λ * MSE(output - pred_mean, target - target_mean)`

#### B. Architecture changes

**B1. Larger latent space**
- Current: 32-dim. If 80% is CT identity (~26 dims), only ~6 dims for per-cell info
- Try 64 or 128 dims — more room for within-CT variation
- Simple to test, low risk

**B2. Disentangled latent: CT code + variation code**
- Split z into z_ct (cell-type identity) + z_var (per-cell variation)
- z_ct: trained with classification loss, used for CT-level imputation
- z_var: trained with reconstruction loss on residuals
- Decoder: takes both, forced to use z_var for per-cell detail
- Similar to beta-VAE disentanglement

**B3. Skip connections from encoder to decoder**
- Let high-frequency per-cell info bypass the bottleneck
- Risk: may leak modality-specific info, breaking cross-decode

#### C. Data/training strategy changes

**C1. Read subsampling augmentation (inspired by low-count cell paper)**
- During training, subsample reads from cells to create "pseudo-nuclei"
- Now we have paired data: (subsampled cell, full cell)
- Train cross-decode on these pairs → explicit per-cell supervision
- This is the most principled approach if we can simulate realistic nucleus-like profiles

**C2. Matching cells across modalities by nearest neighbor in latent space**
- After initial training, find nearest cell-nucleus pairs in z space
- Use these as pseudo-pairs for fine-tuning with per-cell cross-recons loss
- Risk: circular reasoning if z doesn't capture per-cell variation initially

**C3. Progressive training: start with means, then add variation**
- Phase 1: current training (learns CT-level mapping)
- Phase 2: freeze CT-level mapping, fine-tune with variance-preserving loss
- Separates the two objectives temporally

### Priority ranking

**Quick wins (test first):**
1. **B1 (larger latent)** — easiest to test, just change num_latent
2. **A3 (variance regularization)** — simple loss term, directly targets the problem
3. **A4 (residual reconstruction)** — principled, moderate implementation effort

**High potential, more work:**
4. **C1 (read subsampling)** — creates paired data, gold standard approach
5. **B2 (disentangled latent)** — elegant but requires architecture changes
6. **A2 (adversarial)** — powerful but tricky to train

**Mention in talk as future directions:**
- Per-cell variation preservation is an open challenge
- Read subsampling for paired data generation
- Disentangled latent representations
