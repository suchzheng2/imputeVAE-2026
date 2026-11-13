# Training Runs Inventory

## Three Training Routes

All routes aim to train a single shared VAE (vae_nb) that handles both cell and nuclei modalities.
Common params: internal_size=10,9, latent_size=5, batch_size=128, lr=1e-4, subset_ratio=0.2, exclude_mitocarta=True.

---

### Route 1 (Setup 1): cell → nuclei → both

**Stage 1 — cell only** — **completed**
- SLURM job: 928562
- Run folder: `run_2026-03-12-17-29-06_928562`
- Script: `jobs/stage1/train_stage1_cell_v3.sh`
- Data: `10X_cells_v2_v3_AIBS/`, 600 epochs
- Loss: recons + beta*JSD, beta=0.5
- Anneal: kl_anneal=True, start=200, over 400 epochs (JSD is 0 until epoch 200, ramps to 0.5 by epoch 600)

**Stage 2 — fine-tune on nuclei** — **completed**
- SLURM job: 1260842
- Run folder: `run_2026-03-13-00-55-11_1260842`
- Script: `jobs/stage2/train_stage2_cell2nuclei_v3.sh`
- Data: `10X_nuclei_v2_v3_AIBS/`, 600 epochs
- Loaded from: stage1 cell model (928562)
- Loss: recons + beta*JSD, beta=0.5
- Anneal: kl_anneal=True, start=200, over 400 epochs

**Stage 3 — joint with alignment** — **completed** (2026-03-20)
- SLURM job: **2558244** (resubmitted from 2553897)
- Run folder: `run_2026-03-19-23-28-27_4049301`
- Script: `jobs/stage3/train_stage3_cell2nuclei_v3.sh`
- Data: `10X_nuclei_v2_v3_AIBS/`, 600 epochs
- Loaded from: stage1 cell (928562) + stage2 nuclei (1260842)
- Weights:
  - cell_recons=1.0, nuclei_recons=1.0
  - cell_jsd=0.5 (no anneal), nuclei_jsd=0.5 (no anneal)
  - alignment=5.0, anneal from 0 over 100 epochs (linear ramp 0→5.0 at epoch 100)
- Epoch dumps: every 50 epochs (13 dumps: 0,50,...,550,599)
- Final loss: 1.298, alignment_JSD=0.00079
- **Post-training eval:** job 2560652 — **completed** (UMAPs + heatmaps + correlation + gene_corr + scIB)

notes: this setup give too high alignment weight, encoder aligned but decoder ouput corrupted

**Stage 3 — alignment=1.0 rerun** — **completed** (2026-03-22)
- SLURM job: **2136850**
- Run folder: `run_2026-03-22-03-26-46_2136850`
- Data: `10X_nuclei_v2_v3_AIBS/`, 600 epochs
- Loaded from: stage1 cell (928562) + stage2 nuclei (1260842)
- Weights:
  - cell_recons=1.0, nuclei_recons=1.0
  - cell_jsd=0.5 (no anneal), nuclei_jsd=0.5 (no anneal)
  - alignment=1.0, anneal from 0 over 100 epochs (linear ramp 0→1.0 at epoch 100)
  - cross_recons=0.0
- Epoch dumps: every 50 epochs (13 dumps: 0,50,...,550,599)
- Final loss: 1.266, alignment_JSD=0.00833
- Nuclei shift: -0.208, Cell shift: +0.119
- Gene corr (epoch 599): nuc→cell=0.605, nuc→nuc=0.886, cell→cell=0.958
- Correlation: decoded_nuc_vs_raw_cell=0.784, decoded_nuc_vs_raw_nuc=0.991
- Decoder scale: cell mean=1.909, nuclei mean=0.659
- notes: alignment=1.0 (reduced from 5.0) — nuclei shift still bad (-0.208), even worse than align=5.0 (-0.207). Lower alignment doesn't help decoder.
- (duplicate run `run_2026-03-22-00-22-40_2127909` also completed with near-identical results)

---

### Route 2 (Setup 2): nuclei → cell → both

**Stage 1 — nuclei only** — **completed**
- SLURM job: 928561
- Run folder: `run_2026-03-12-17-29-06_928561`
- Script: `jobs/stage1/train_stage1_nuclei_v3.sh`
- Data: `10X_nuclei_v2_v3_AIBS/`, 600 epochs
- Loss: recons + beta*JSD, beta=0.5
- Anneal: kl_anneal=True, start=200, over 400 epochs

**Stage 2 — fine-tune on cell** — **completed**
- SLURM job: 998747
- Run folder: `run_2026-03-13-01-05-55_998747`
- Script: `jobs/stage2/train_stage2_nuclei2cell_v3.sh`
- Data: `10X_cells_v2_v3_AIBS/`, 600 epochs
- Loaded from: stage1 nuclei model (928561)
- Loss: recons + beta*JSD, beta=0.5
- Anneal: kl_anneal=True, start=200, over 400 epochs

**Stage 3 — joint with alignment** — **completed** (2026-03-20)
- SLURM job: **2558245** (resubmitted from 2553898)
- Run folder: `run_2026-03-20-00-00-32_2930615`
- Script: `jobs/stage3/train_stage3_nuclei2cell_v3.sh`
- Data: `10X_cells_v2_v3_AIBS/`, 600 epochs
- Loaded from: stage1 nuclei (928561) + stage2 cell (998747)
- Weights:
  - cell_recons=1.0, nuclei_recons=1.0
  - cell_jsd=0.5 (no anneal), nuclei_jsd=0.5 (no anneal)
  - alignment=5.0, anneal from 0 over 100 epochs (linear ramp 0→5.0 at epoch 100)
- Epoch dumps: every 50 epochs (13 dumps: 0,50,...,550,599)
- Final loss: 1.298, alignment_JSD=0.00107
- **Post-training eval:** job 2560653 — **completed** (UMAPs + heatmaps + correlation + gene_corr + scIB)

notes: this setup give too high alignment weight, encoder aligned but decoder ouput corrupted

**Stage 3 — alignment=1.0 rerun** — **completed** (2026-03-22)
- SLURM job: **2119158**
- Run folder: `run_2026-03-21-21-23-54_2119158`
- Data: `10X_cells_v2_v3_AIBS/`, 600 epochs
- Loaded from: stage1 nuclei (928561) + stage2 cell (998747)
- Weights:
  - cell_recons=1.0, nuclei_recons=1.0
  - cell_jsd=0.5 (no anneal), nuclei_jsd=0.5 (no anneal)
  - alignment=1.0, anneal from 0 over 100 epochs (linear ramp 0→1.0 at epoch 100)
  - cross_recons=0.0
- Epoch dumps: every 50 epochs (13 dumps: 0,50,...,550,599)
- Final loss: 1.271, alignment_JSD=0.0326
- Nuclei shift: -0.195, Cell shift: +0.106
- Gene corr (epoch 599): nuc→cell=0.591, nuc→nuc=0.951, cell→cell=0.963
- Correlation: decoded_nuc_vs_raw_cell=0.801, decoded_nuc_vs_raw_nuc=0.996
- Decoder scale: cell mean=1.910, nuclei mean=0.658
- notes: alignment=1.0 (reduced from 5.0) — nuclei shift still poor (-0.195). Route 2 has slightly worse alignment convergence (JSD=0.033 vs 0.008 for Route 1), but slightly better nuc→nuc gene corr (0.951 vs 0.886).

---

### Route 3 (Setup 3): cell → both (skip separate stage 2)

**Stage 1 — cell only** (same as Route 1)
- Run folder: `run_2026-03-12-17-29-06_928562` — **completed**

**Stage 2/3 — joint cell2both**

*Full run, no alignment anneal (600 epochs)* — **completed** (2026-03-20)
- SLURM job: **2558243** (resubmitted from 2551849)
- Run folder: `run_2026-03-19-23-28-27_1980580`
- Script: `jobs/stage3/train_stage2_cell2both_v3.sh`
- Data: `10X_nuclei_v2_v3_AIBS/`, 600 epochs
- Loaded from: stage1 cell (928562), no stage2 model
- Weights:
  - cell_recons=1.0, nuclei_recons=1.0
  - cell_jsd=0.5 (no anneal), nuclei_jsd=0.5 (no anneal)
  - alignment=5.0, **no annealing** (full weight from epoch 0)
- Epoch dumps: every 50 epochs (13 dumps: 0,50,...,550,599)
- Final loss: 1.302, alignment_JSD=0.00060
- Final shift: nuclei=-0.116, cell=-0.099
- Final nuclei→cell corr: 0.674
- **Post-training eval:** job 2560651 — **completed** (UMAPs + heatmaps + correlation + gene_corr + scIB)
- notes: this setup give too high alignment weight, encoder aligned but decoder ouput corrupted

*Freq eval run (100 epochs)* — **completed** (2026-03-20)
- SLURM job: **2557801**
- Run folder: `run_2026-03-19-22-50-33_2343862`
- Script: `jobs/stage3/train_stage2_cell2both_v3_freq_eval.sh`
- Data: `10X_nuclei_v2_v3_AIBS/`, 100 epochs, eval every 10, dump every 10
- Same weights as full run above
- **Post-training eval:** job 2558785 — **completed** (UMAPs + heatmaps + correlation + gene_corr)

*Cell2both + cross-recons (600 epochs)* — **completed** (2026-03-20)
- SLURM job: **2560805**
- Run folder: `run_2026-03-20-09-18-17_2009562`
- Script: `jobs/stage3/train_stage2_cell2both_v3_cross_recons.sh`
- Data: `10X_nuclei_v2_v3_AIBS/`, 600 epochs
- Loaded from: stage1 cell (928562), no stage2 model
- Weights: same as full run above, plus:
  - cross_recons=0.5, annealed from 0 over 100 epochs
  - alignment=5.0, no annealing
- Final loss: 1.698, alignment_JSD=0.00050, cross_recons_loss=0.790
- Final shift: nuclei=-0.071, cell=-0.070
- Final nuclei→cell corr: 0.762

*Cell2both + annealed alignment baseline (600 epochs)* — **completed** (2026-03-20)
- SLURM job: **2560882**
- Run folder: `run_2026-03-20-09-38-40_409694`
- Script: `jobs/stage3/train_stage2_cell2both_v3_anneal.sh`
- Data: `10X_nuclei_v2_v3_AIBS/`, 600 epochs
- Loaded from: stage1 cell (928562), no stage2 model
- Weights: same as full run above, but:
  - alignment=5.0, annealed from 0 over 100 epochs (linear ramp)
  - cross_recons=0.0 (disabled)
- Final loss: 1.334, alignment_JSD=0.01315
- Final shift: nuclei=-0.123, cell=-0.033
- Final nuclei→cell corr: 0.717

**Previous attempt (FAILED)**
- Run folder: `run_2026-03-13-00-55-52_997499` — 600 epochs completed, alignment never converged
- Old weights:
  - cell_recons=2.0, nuclei_recons=1.0
  - cell_jsd=1.0, nuclei_jsd=1.0
  - alignment=0.5 (alpha), kl_anneal=True, start=200, over 400 epochs
- **Why it failed:** alignment weight (0.5) too weak vs recons (1.0-2.0), delayed start let model settle into unaligned minimum

---

### Quick Probes (2026-03-20): Weight Strategy Exploration — ALL COMPLETED

100 epochs, subset_ratio=0.05, eval every 25 epochs, no epoch dumps.
All use stage1 cell model (928562), cell2both (Setup 3 route).
Scripts: `jobs/probe/probe_{A,B,C,D,E}_*.sh`

#### Probe Rationale

- **Probe A (baseline control):** Identical weights to prior full runs (alignment=5.0, no anneal, no cross-recons). Serves as the reference point to measure improvement. Any difference from full runs reflects the smaller data (subset_ratio=0.05 vs 0.2).

- **Probe B (relax nuclei faithfulness):** Hypothesis: if the decoder is too strongly trained to reproduce nuclei-specific patterns (nuclei_recons=1.0), it can't shift toward cell-like output. By lowering nuclei_recons to 0.3, we give the decoder more freedom. Alignment reduced to 3.0 (annealed) since lower reconstruction makes alignment relatively stronger.

- **Probe C (bias decoder toward cell):** Hypothesis: if cell reconstruction dominates (cell_recons=2.0), the shared decoder learns more cell-like internal representations, which may bleed into nuclei decoding. nuclei_recons lowered to 0.5 to create asymmetry. Tests whether indirect decoder bias via weight imbalance is enough.

- **Probe D (cross-recons, direct approach):** Tests the new cross-modality reconstruction loss in isolation. All original reconstruction weights kept at 1.0 — the only change is adding cross_recons=1.0 (annealed) which directly tells the decoder "decoded nuclei should look like raw cell." Alignment reduced to 3.0 since cross-recons provides additional cross-modality signal.

- **Probe E (combined strategy):** Combines the two most promising ideas: cross-recons (0.5) + lowered nuclei_recons (0.5). Lighter alignment (2.0) to avoid over-constraining. The idea is that relaxing nuclei faithfulness AND adding direct cell target pressure will produce the strongest shift.

#### Configuration

| Probe | Job | Run Folder | nuc_recons | cell_recons | alignment | cross_recons | anneal |
|-------|-----|-----------|-----------|-------------|-----------|-------------|--------|
| A | 2563189 | `run_2026-03-20-10-27-02_2073068` | 1.0 | 1.0 | 5.0 | 0.0 | none |
| B | 2563190 | `run_2026-03-20-10-57-36_2707095` | 0.3 | 1.0 | 3.0→50ep | 0.0 | alignment |
| C | 2563191 | `run_2026-03-20-10-26-23_2685978` | 0.5 | 2.0 | 3.0→50ep | 0.0 | alignment |
| D | 2563192 | `run_2026-03-20-10-27-02_2958891` | 1.0 | 1.0 | 3.0→50ep | 1.0→50ep | both |
| E | 2563193 | `run_2026-03-20-10-27-02_3292804` | 0.5 | 1.0 | 2.0→50ep | 0.5→50ep | both |

#### Results (epoch 99)

**Shift scores** (higher = more cell-like; positive means decoded nuclei resemble cells):

| Probe | Nuclei Shift | Cell Shift | Interpretation |
|-------|-------------|-----------|----------------|
| A baseline | -0.133 | -0.119 | Same as prior full runs — decoder preserves modality |
| B low nuclei | -0.069 | -0.046 | 2x better, relaxing nuclei helps modestly |
| C boost cell | -0.164 | +0.072 | Worst for nuclei — boosting cell recons doesn't help nuclei |
| **D cross-recons** | **-0.040** | -0.007 | 3x better — cross-recons is the key ingredient |
| **E combined** | **-0.020** | +0.026 | Best — nearly neutral, closest to cell-like |

**Decoded nuclei vs raw cell/nuclei correlation** (gene-level):

| Probe | Nuclei→Cell corr | Nuclei→Nuclei corr |
|-------|-----------------|-------------------|
| A baseline | 0.608 | 0.653 |
| B low nuclei | 0.736 | 0.632 |
| C boost cell | 0.659 | 0.628 |
| **D cross-recons** | **0.841** | 0.764 |
| **E combined** | **0.833** | 0.704 |

**Alignment JSD** (latent convergence):

| Probe | Alignment JSD |
|-------|--------------|
| A | 0.002 |
| B | 0.003 |
| C | 0.020 |
| D | 0.005 |
| E | 0.008 |

#### Key Findings

1. **Cross-recons is the critical ingredient.** D and E both use it and dramatically outperform A/B/C.
2. **Weight rebalancing alone (B, C) is insufficient.** Lowering nuclei_recons or boosting cell_recons helps modestly but doesn't solve the problem.
3. **D vs E are very close.** D has slightly better nuclei→cell corr (0.841 vs 0.833) and tighter alignment (0.005 vs 0.008). E has better shift score (-0.020 vs -0.040). Both are strong candidates for full 600-epoch runs.
4. **C is counterproductive for nuclei.** Boosting cell_recons makes cells better but actually worsens nuclei shift (-0.164).

---

### Cross-Recons Weight Sweep (2026-03-23): Probes F1-F6 — ALL COMPLETED

100 epochs, subset_ratio=0.1, eval every 50 epochs, cell2both (Route 3).
All use stage1 cell model (928562). Full eval enabled (scIB at epochs 0, 50, 99).
cell_recons=1.0, cell_jsd=0.5, nuclei_jsd=0.5 throughout.
Original jobs (2774195–2774200) were preempted; resubmitted as jobs 2776905–2776910, all completed.

**Goal:** Find sweet spot between cross_recons (push nuclei→cell distribution) and nuclei_recons (preserve individual variation). Distribution-level matching, not cell-level.

#### Configuration

| Probe | Job | Run Folder | cross_recons | nuclei_recons | alignment | anneal |
|-------|-----|-----------|-------------|---------------|-----------|--------|
| F1 | 2776905 | `run_2026-03-23-16-41-49_84803` | 0.5→50ep | 1.0 | 3.0→50ep | both |
| F2 | 2776906 | `run_2026-03-23-16-42-49_2436696` | 1.0→50ep | 1.0 | 3.0→50ep | both |
| F3 | 2776907 | `run_2026-03-23-16-45-43_2254901` | 2.0→50ep | 1.0 | 3.0→50ep | both |
| F4 | 2776908 | `run_2026-03-23-16-45-43_2699264` | 0.5→50ep | 0.5 | 2.0→50ep | both |
| F5 | 2776909 | `run_2026-03-23-17-18-17_2701118` | 1.0→50ep | 0.5 | 2.0→50ep | both |
| F6 | 2776910 | `run_2026-03-23-17-25-49_2256956` | 2.0→50ep | 0.5 | 2.0→50ep | both |

**Key comparisons:**
- F1 vs F2 vs F3: does increasing cross_recons help or collapse variation? (nuclei_recons=1.0)
- F4 vs F5 vs F6: same question with relaxed nuclei_recons=0.5
- F1 vs F4, F2 vs F5, F3 vs F6: does lowering nuclei_recons help at each cross_recons level?

#### Results — Shift Scores (epoch 99)

| Probe | Train Nuc Shift | Train Cell Shift | Valid Nuc Shift | Valid Cell Shift |
|-------|----------------|-----------------|----------------|-----------------|
| F1 | -0.103 | -0.074 | -0.102 | -0.055 |
| F2 | -0.031 | -0.034 | -0.073 | -0.014 |
| F3 | +0.010 | +0.022 | -0.023 | +0.036 |
| F4 | -0.041 | -0.030 | -0.059 | -0.001 |
| F5 | +0.001 | +0.036 | -0.038 | +0.017 |
| **F6** | **+0.060** | **+0.084** | **-0.005** | **+0.044** |

#### Results — Gene Correlation (epoch 99)

| Probe | Train Cell→Cell | Train Nuc→Nuc | Train Cell→Nuc | Valid Cell→Cell | Valid Nuc→Nuc | Valid Cell→Nuc |
|-------|----------------|--------------|---------------|----------------|--------------|---------------|
| F1 | 0.940 | 0.763 | 0.673 | 0.906 | 0.712 | 0.614 |
| F2 | 0.930 | 0.760 | 0.662 | 0.919 | 0.715 | 0.611 |
| F3 | 0.968 | 0.753 | 0.642 | 0.942 | 0.720 | 0.627 |
| F4 | 0.947 | 0.720 | 0.619 | 0.919 | 0.691 | 0.577 |
| F5 | 0.971 | 0.722 | 0.625 | 0.934 | 0.699 | 0.567 |
| **F6** | **0.977** | 0.716 | 0.602 | **0.950** | 0.665 | 0.558 |

#### Results — Correlation Matrix: Decoded Nuclei vs Raw (epoch 99)

| Probe | Train dec_nuc↔raw_cell | Train dec_nuc↔raw_nuc | Valid dec_nuc↔raw_cell | Valid dec_nuc↔raw_nuc |
|-------|----------------------|---------------------|----------------------|---------------------|
| F1 | 0.874 | 0.977 | 0.870 | 0.972 |
| F2 | 0.918 | 0.949 | 0.886 | 0.959 |
| F3 | 0.936 | 0.927 | 0.917 | 0.939 |
| F4 | 0.912 | 0.953 | 0.899 | 0.957 |
| F5 | 0.936 | 0.935 | 0.907 | 0.946 |
| **F6** | **0.959** | 0.900 | **0.925** | 0.929 |

#### Results — Decoder Scale (epoch 99)

| Probe | Train Cell Mean | Train Nuc Mean | Valid Cell Mean | Valid Nuc Mean |
|-------|----------------|---------------|----------------|---------------|
| F1 | 1.868 | 0.644 | 1.867 | 0.646 |
| F2 | 1.893 | 0.647 | 1.867 | 0.646 |
| F3 | 1.871 | 0.646 | 1.867 | 0.646 |
| F4 | 1.882 | 0.645 | 1.867 | 0.646 |
| F5 | 1.876 | 0.646 | 1.867 | 0.646 |
| F6 | 1.870 | 0.645 | 1.867 | 0.646 |

#### Results — Loss Components (epoch 99)

| Probe | Train Total | Valid Total | Train Align JSD | Valid Align JSD | Train Cross Recons | Valid Cross Recons |
|-------|------------|------------|----------------|----------------|-------------------|-------------------|
| F1 | 1.694 | 1.707 | 0.0030 | 0.0061 | 0.778 | 0.779 |
| F2 | 2.092 | 2.100 | 0.0036 | 0.0074 | 0.777 | 0.776 |
| F3 | 2.844 | 2.874 | 0.0045 | 0.0087 | 0.764 | 0.771 |
| F4 | 1.435 | 1.441 | 0.0036 | 0.0061 | 0.772 | 0.774 |
| F5 | 1.818 | 1.831 | 0.0050 | 0.0073 | 0.766 | 0.771 |
| F6 | 2.575 | 2.602 | 0.0062 | 0.0097 | 0.759 | 0.767 |

#### Key Findings

1. **Higher cross_recons consistently improves cell-likeness** — within both nuc_recons groups, increasing cross_recons (0.5→1.0→2.0) monotonically improves shift scores and dec_nuc↔raw_cell correlation.
2. **Lowering nuclei_recons (1.0→0.5) provides additional shift improvement** but sacrifices nuclei fidelity (nuc→nuc gene corr drops ~0.03–0.05).
3. **F6 is the strongest cell-like converter** — train nuc shift +0.060 (valid -0.005), best dec_nuc↔raw_cell (0.959/0.925), but nuc→nuc fidelity drops to 0.716/0.665.
4. **F3 is the best balanced option** — near-neutral shift (+0.010/-0.023), strong dec_nuc↔raw_cell (0.936/0.917), retains higher nuc→nuc (0.753/0.720) thanks to full nuclei_recons=1.0.
5. **No overfitting observed** — train and valid metrics track closely across all probes.
6. **Decoder scale is invariant** — cell mean ~1.87, nuclei mean ~0.65 across all configurations, confirming scale is set by the pretrained decoder, not by loss weights.
7. **Total loss scales with cross_recons weight** (as expected) but does not indicate model quality — F6 has highest loss but best shift.

---

### Alignment Weight Sweep (2026-03-20): No Cross-Recons — ALL COMPLETED

300 epochs, subset_ratio=0.1, eval every 50 epochs, cell2both (Setup 3 route).
All use stage1 cell model (928562), alignment annealed over 100 epochs, no cross-recons.

#### Configuration

| Run Folder | alignment | nuc_jsd | cell_jsd | nuc_recons | cell_recons |
|------------|-----------|---------|----------|-----------|-------------|
| `run_2026-03-20-16-13-55_2032626` | 0.5→100ep | 0.5 | 0.5 | 1.0 | 1.0 |
| `run_2026-03-20-16-25-35_3327327` | 1.0→100ep | **0.0** | 0.5 | 1.0 | 1.0 |
| `run_2026-03-20-16-25-35_2832493` | 2.0→100ep | 0.5 | 0.5 | 1.0 | 1.0 |

#### Results (epoch 299)

| Run | Align wt | Nuclei Shift | Nuc→Cell corr | Nuc→Nuc corr | Cell→Cell corr |
|-----|----------|-------------|---------------|--------------|----------------|
| `2032626` | 0.5 | -0.210 | 0.677 | 0.877 | 0.963 |
| `3327327` | 1.0 (no nuc JSD) | -0.205 | 0.722 | 0.958 | 0.974 |
| `2832493` | 2.0 | -0.218 | 0.664 | 0.847 | 0.883 |

#### Key Findings

1. **Alignment weight (0.5→2.0) has no effect on nuclei shift** — all three at ~-0.21, even worse than 600ep runs with align=5.0 (-0.12). Confirms alignment only acts in latent space, not decoder.
2. **Removing nuclei JSD (`3327327`, nuc_jsd=0.0) helps gene correlation** — best nuc→cell corr (0.722) and nuc→nuc corr (0.958), but shift is still terrible (-0.205).
3. **Higher alignment hurts cell reconstruction** — align=2.0 drops cell→cell corr to 0.883 (vs 0.963 at align=0.5) without improving nuclei shift.
4. **300 vs 600 epochs doesn't matter without cross-recons.** Comparing to the 600ep full runs: nuclei shift plateaus by ~epoch 100-150 and oscillates thereafter. The baseline 600ep run (align=5.0) went from shift=-0.098 at epoch 300 to -0.116 at epoch 599 (no improvement). The annealed 600ep run went from -0.127 to -0.123. Extra epochs provide no benefit when the decoder lacks a cross-modality gradient signal.
5. **Cross-recons is the only run type where later epochs help** — the cross-recons 600ep run improved from shift=-0.097 (ep300) to -0.071 (ep599) and nuc→cell corr from 0.729 to 0.762.

---

### Conditioned Decoder — Delayed Alignment Experiments (2026-03-26)

**Context:** Run `run_2026-03-26-14-42-54_337054` (conditioned decoder probe, 100 epochs from stage1 cell) showed good latent alignment but nuclei reconstruction was not good enough (0.82 corr to raw nuclei vs 0.95 for cell). Cross-reconstruction (nuclei with cell decoder head) looked good (0.936 corr to raw cell), confirming the decoder capacity is there but the nuclei embedding doesn't get enough signal.

**Hypothesis:** Strong alignment (weight=3.0, annealed from epoch 0) pushes nuclei latents to be cell-like too early, before the nuclei decoder head has learned proper nuclei reconstruction. By delaying alignment to epoch 100, the nuclei decoder head can first learn to reconstruct nuclei well, then alignment is introduced gradually.

**Strategy:** No alignment for first 100 epochs (pure reconstruction), then ramp alignment 0→1.5 over epochs 100–149, steady at 1.5 for epochs 150–299. No cross-recons. No nuclei_recons_weight boost — isolate the effect of delayed alignment.

#### Run A: From stage1 cell model — **submitted** (2026-03-26)
- SLURM job: **2886168**
- Script: `jobs/stage3/train_cond_delayed_align.sh`
- Init model: stage1 cell (`run_2026-03-12-17-29-06_928562`)
- 300 epochs, subset_ratio=0.1, eval every 25
- Weights: cell_recons=1.0, cell_jsd=0.5, nuclei_recons=1.0, nuclei_jsd=0.5
- alignment_weight=1.5, anneal from epoch 100 over 50 epochs
- embed_dim=8
- **Purpose:** Test whether delayed alignment lets the nuclei decoder head learn better reconstruction before alignment pressure distorts the latent space. Baseline comparison to run_337054 which had alignment from epoch 0.

#### Run B: From stage2 cell2nuclei model — **submitted** (2026-03-26)
- SLURM job: **2886166**
- Script: `jobs/stage3/train_cond_delayed_align_from_stage2.sh`
- Init model: stage2 cell2nuclei (`run_2026-03-13-00-55-11_1260842`)
- 300 epochs, subset_ratio=0.1, eval every 25
- Weights: same as Run A
- alignment_weight=1.5, anneal from epoch 100 over 50 epochs
- embed_dim=8
- **Purpose:** The stage2 model already knows both cell and nuclei reconstruction. Starting from a better init should give the conditioned decoder a head start — the modality embeddings can specialize on top of an encoder that already separates modalities. Compare to Run A to see if better init improves final nuclei reconstruction quality.

---

---

### V3 MSE vs NB Comparison Runs (2026-03-31)

**Context:** All previous runs used vae_nb (NB reconstruction loss on raw counts). We want to compare
against a simpler MSE loss on log1p-normalized data to see whether the NB formulation is actually
helping or if the softmax bottleneck in NB is hurting. Also, all previous runs used the old gene
filtering (detection_rate_threshold=1.0 → ~24.7K genes). These new runs use Config B filtering
(det>5%, exclude_mitocarta → ~13-14K genes), which removes ~10K noise genes.

**Changes from previous runs:**
1. **Config B gene filtering**: detection_rate_threshold=5.0, exclude_mitocarta=True
2. **New training schedule**: stage 1 = 600 epochs with recons-only for 300, then JSD anneal 300-400, then both 400-600
3. **Stage 2 = code's stage3_v2** with cascading loss annealing: cell always on, nuclei_recons ramps 0-100, nuclei_jsd delayed to 100 (ramp 100-200), alignment delayed to 200 (ramp 200-300)
4. **Two model types side by side**: vae_js (MSE on log1p) vs vae_nb (NB on raw counts)

#### MSE Run: Stage 1 — cell only (vae_js) — **completed** (2026-04-01)
- SLURM jobs: 3017418 (failed: missing --batch, --use_raw_data, OOM) → 3018209 (failed: stale __pycache__) → **3057683** (completed)
- Run folder: `run_2026-04-01-09-29-26_3250320`
- Script: `jobs/run_s1_cell_vae_js_v3.sh`
- Data: `10X_cells_v2_v3_AIBS/`, batch=c3A, protocol=v3
- Model: `vae_js`, recons_loss_type=0 (MSE), use_raw_data=True, raw_data_normalization=log1p
- Architecture: internal_size=10,9, latent_size=5, **10,134 genes** (Config B gene list)
- 600 epochs, batch_size=128, lr=1e-4, beta=0.5, mem=400G
- Training data: 71,117 cells (after v3 filter + 0.2 subset)
- JSD anneal: start=300, over 100 epochs
- Final epoch 599 loss: **~0.275** (recons ~0.250, KL ~0.049)
- Duration: 3h 39m, UMAPs all generated

#### NB Run: Stage 1 — cell only (vae_nb) — **completed** (training), UMAPs pending (2026-04-01)
- SLURM jobs: 3017450 (failed: missing --batch, OOM) → 3018211 (failed: stale __pycache__) → **3057684** (preempted at epoch 491 during UMAP gen, but all 600 epochs trained)
- Run folder: `run_2026-04-01-09-29-27_2189144`
- Continue job: **3073910** (from epoch 599 — just final eval + UMAPs)
- Script: `jobs/run_s1_cell_vae_nb_v3.sh`
- Data: `10X_cells_v2_v3_AIBS/`, batch=c3A, protocol=v3
- Model: `vae_nb`, raw counts (use_raw_data=True, raw_data_normalization=raw)
- Architecture: same as MSE, **10,134 genes** (Config B gene list)
- 600 epochs, batch_size=128, lr=1e-4, beta=0.5, mem=400G
- Training data: 71,117 cells (after v3 filter + 0.2 subset)
- JSD anneal: start=300, over 100 epochs
- Final epoch 598 loss: **~1.991** (recons ~1.646, KL ~0.689)
- Note: epoch dumps saved through 599, model.pt saved; preemption hit during post-training UMAP at epoch 450. Missing: epoch 599 loss line, UMAPs for epochs 500+

#### MSE Run: Stage 2 — joint cell+nuclei (vae_js) — **completed** (2026-04-01)
- SLURM jobs: 3018349 (cancelled, dep on failed stage1) → **3073916** (completed, 4h 11m)
- Run folder: `run_2026-04-01-14-01-16_1946018`
- Script: `jobs/run_s2_joint_vae_js_v3.sh`
- Stage 1 model: `run_2026-04-01-09-29-26_3250320/models/model.pt`
- Data: data_path=nuclei (target=nuclei, reference=cell), batch=n3A
- Model: `vae_js`, use_raw_data=True, raw_data_normalization=log1p
- 600 epochs, mem=400G, time=24h
- Loss schedule:
  - cell_recons=1.0 (always on), cell_jsd=0.5 (always on)
  - nuclei_recons=1.0, anneal 0→100
  - nuclei_jsd=0.5, delayed 100, anneal 100→200
  - alignment=1.0, delayed 200, anneal 200→300
  - cross_recons=0.0 (off)
- Eval every 50, epoch dumps every 50

#### NB Run: Stage 2 — joint cell+nuclei (vae_nb) — **completed** (2026-04-01)
- SLURM jobs: 3018350 (cancelled, dep on failed stage1) → **3077244** (completed, 2h 48m)
- Run folder: `run_2026-04-01-14-43-14_475265`
- Script: `jobs/run_s2_joint_vae_nb_v3.sh`
- Stage 1 model: `run_2026-04-01-09-29-27_2189144/models/model.pt`
- Same as MSE stage 2 but with vae_nb model, raw counts
- **Post-training eval:** job 3104562 (preempted, partial) → **3106114** (completed, 35m)

---

### Conditioned Decoder Stage 2 — MSE & NB (2026-04-02)

**Context:** Previous conditioned decoder experiment (run_337054, 100 epochs) used aggressive alignment (0→3.0 over 100 epochs) and showed nuclei recon degradation. The baseline (shared decoder) stage 2 uses cascading annealing with alignment delayed to epoch 200. These runs apply the same cascading schedule to the conditioned decoder architecture for a fair comparison.

**Architecture:** `VAE_JSD_Cond` (new, MSE) and `VAE_NB_Cond` (existing, NB) — shared encoder with modality embedding (embed_dim=8) concatenated to z before decoding. Lets decoder specialize per modality while sharing latent space.

**Code changes for this experiment:**
- Added `VAE_JSD_Cond` class to `imputeVAE/models_cond.py` (mirrors `VAE_NB_Cond` but extends `VAE_JSD`, keeps final LeakyReLU)
- Updated `cond_training.py` to dispatch on `args.model` (`vae_js` → `VAE_JSD_Cond`, `vae_nb` → `VAE_NB_Cond`)

#### MSE Conditioned Decoder: Stage 2 — **completed** (2026-04-02)
- SLURM job: **3102281** (completed, 6h 16m)
- Script: `jobs/run_s2_cond_vae_js_v3.sh`
- Stage 1 model: `run_2026-04-01-09-29-26_3250320/models/model.pt` (vae_js cell, Config B)
- Model: `VAE_JSD_Cond`, embed_dim=8, use_raw_data=True, raw_data_normalization=log1p
- Data: data_path=nuclei (target=nuclei, reference=cell), batch=n3A, protocol=v3
- Architecture: internal_size=10,9, latent_size=5, 10,134 genes (Config B gene list)
- 600 epochs, batch_size=128, lr=1e-4, beta=0.5, mem=400G
- Loss schedule (same as baseline stage 2):
  - cell_recons=1.0 (always on), cell_jsd=0.5 (always on)
  - nuclei_recons=1.0, anneal 0→100
  - nuclei_jsd=0.5, delayed 100, anneal 100→200
  - alignment=1.0, delayed 200, anneal 200→300
  - cross_recons=0.0 (off)
- Eval every 50, epoch dumps every 50
- **Purpose:** Compare conditioned decoder (modality embedding) vs shared decoder (baseline stage 2 job 3073916) for MSE model. Expect better nuclei reconstruction since decoder can specialize.
- **Post-training eval:** job 3106115 (preempted at 14m) → **3115974** (completed, 35m)
- **Eval results (epoch 599):** cell corr=0.9117, nuclei→cell corr=0.7556, best nuclei→cell=0.7831 at epoch 0, scale std=0.547
- **Verdict:** Nearly identical to MSE shared (0.76 vs 0.76). Cond decoder doesn't help without cross-recons.

#### NB Conditioned Decoder: Stage 2 — **completed** (2026-04-02)
- SLURM job: **3102282** (completed, 4h 12m)
- Script: `jobs/run_s2_cond_vae_nb_v3.sh`
- Stage 1 model: `run_2026-04-01-09-29-27_2189144/models/model.pt` (vae_nb cell, Config B)
- Model: `VAE_NB_Cond`, embed_dim=8, use_raw_data=True, raw_data_normalization=raw
- Same architecture, same loss schedule as MSE run above
- **Purpose:** Same comparison for NB model. Also compare MSE-cond vs NB-cond head-to-head.
- **Post-training eval:** job 3106116 (preempted at 24m) → **3115975** (completed, 35m)
- **Eval results (epoch 599):** cell corr=0.8727, nuclei→cell corr=0.6884, best nuclei→cell=0.7060 at epoch 400, scale std=21.66
- **Verdict:** Worse than NB shared (0.69 vs 0.73). Cond decoder hurts NB without cross-recons. Scale distortion confirmed (std=21.7).

#### Comparisons planned:
1. **Cond vs shared decoder (MSE):** job 3102281 vs job 3073916 — does modality embedding prevent decoder degradation?
2. **Cond vs shared decoder (NB):** job 3102282 vs job 3077244 — same question for NB
3. **MSE-cond vs NB-cond:** job 3102281 vs 3102282 — which loss works better with conditioned decoder?
4. **Cross-decode quality:** conditioned decoder can cross-decode (nuclei latent + cell embedding) — compare to shared decoder nuclei→cell output

---

### Conditioned Decoder + Cross-Recons — MSE (2026-04-02)

**Context:** Two separate problems identified in earlier experiments:
1. **Decoder degradation** — shared decoder pulled by competing modality losses → solved by conditioned decoder (modality embedding)
2. **No direct nuclei→cell supervision** — alignment works in latent space but doesn't translate to decoder output → solved by cross-recons

Previous cross-recons experiments (Probes D-F, shared decoder NB) showed cross-recons was the *key ingredient* for shift scores (Probe D: 3x better than baseline). But those used the shared decoder, which applies the same decoder to both modalities. The conditioned decoder enables a cleaner cross-recons:
- Encode nuclei → z_nuc
- Decode z_nuc with **cell embedding** → cross-decoded output
- Loss = MSE(cross-decoded, raw_cell)

This trains the decoder's cell head to produce cell-like output from nuclei latents, without distorting the encoder (mu is detached before cross-decode).

**Code changes:**
- Added `decode_cross_train()` to `VAE_JSD_Cond` and `VAE_NB_Cond` in `models_cond.py` — like `decode_cross` but with gradients, mu detached so only decoder+embedding get gradient
- Added cross-recons loss computation to `cond_training.py` training loop and validation loop
- For MSE: `mse_loss(cross_decoded, cell_data)` — both in log1p space
- For NB: NB log-likelihood of raw cell counts given cross-decoded mu (with cell library size)
- Cross-recons weight uses same annealing infrastructure as other weights

#### MSE Cond + Cross-Recons: Stage 2 — **running** (epoch ~346/600) (2026-04-02)
- SLURM job: **3106445** (on scavenge_gpu)
- Script: `jobs/run_s2_cond_crossrecons_vae_js_v3.sh`
- Stage 1 model: `run_2026-04-01-09-29-26_3250320/models/model.pt` (vae_js cell, Config B)
- Model: `VAE_JSD_Cond`, embed_dim=8, raw_data_normalization=log1p
- Data: data_path=nuclei (target=nuclei, reference=cell), batch=n3A, protocol=v3
- Architecture: internal_size=10,9, latent_size=5, 10,134 genes (Config B gene list)
- 600 epochs, batch_size=128, lr=1e-4, beta=0.5, mem=400G
- Loss schedule:
  - cell_recons=1.0 (always on), cell_jsd=0.5 (always on)
  - nuclei_recons=1.0, anneal 0→100
  - cross_recons=1.0, delayed 100, anneal 100→200
  - nuclei_jsd=0.5, delayed 100, anneal 100→200
  - alignment=1.0, delayed 200, anneal 200→300
- **Key difference from job 3102281:** adds cross_recons=1.0 (annealed from epoch 100)
- **Purpose:** Test whether direct nuclei→cell supervision via conditioned decoder cross-recons improves shift scores and cross-modal reconstruction quality. This combines our two best architectural ideas.

#### Comparisons planned:
1. **Cond+xr vs Cond-only (MSE):** job 3106445 vs job 3102281 — does cross-recons improve shift?
2. **Cond+xr vs shared decoder baseline:** job 3106445 vs job 3073916 — overall improvement?
3. **Cond+xr MSE vs old Probe F3 (shared NB+xr):** best cross-recons config vs best of the old shared decoder probes

---

### MSE No-Filter Comparison Run (2026-04-02)

**Context:** All new runs use Config B gene filtering (10,134 genes). To isolate the effect of gene filtering vs loss function, we need an MSE run with the same ~24.7K genes as the old NB runs.

#### MSE No-Filter: Stage 1 (cell only) — **resubmitted** (2026-04-02)
- SLURM job: **3106245** (OOM at 400G after 8min) → **3115964** (resubmitted with 500G)
- Script: `jobs/run_s1_cell_vae_js_v3_nofilter.sh`
- Model: vae_js (MSE), use_raw_data=True, raw_data_normalization=log1p
- Data: 10X_cells_v2_v3_AIBS/, batch=c3A, protocol=v3
- Gene filtering: detection_rate_threshold=1.0, exclude_mitocarta=True (~24.7K genes, same as old NB runs)
- 600 epochs, JSD anneal start=300 over 100, batch_size=128, lr=1e-4
- **Purpose:** Fill in the 2×2 comparison matrix (MSE vs NB × filter vs no-filter)
- **Note:** 24.7K genes OOM'd at 400G during data loading. Bumped to 500G.

#### MSE No-Filter: Stage 2 — **script ready, waiting for stage 1**
- Script: `jobs/run_s2_joint_vae_js_v3_nofilter.sh`
- STAGE1_MODEL placeholder needs update after stage 1 completes
- Same cascading loss schedule as other stage 2 runs

#### 2×2 Comparison Matrix (all stage 2, v3 only)

| | No filter (~24.7K genes) | Config B (10,134 genes) |
|---|---|---|
| **MSE** | job 3106245 (stage 1 running) | completed: s1=3250320, s2=1946018 |
| **NB** | old runs: s1=928562, s2=various | completed: s1=2189144, s2=475265 |

---

## Stage 1 Models

| Run Folder | Modality | Model | Epochs | beta | JSD anneal | Gene filter |
|------------|----------|-------|--------|------|------------|-------------|
| `run_2026-03-12-17-29-06_928562` | cell | vae_nb | 600 | 0.5 | start=200, over 400 | det>1% (~24.7K) |
| `run_2026-03-12-17-29-06_928561` | nuclei | vae_nb | 600 | 0.5 | start=200, over 400 | det>1% (~24.7K) |
| `run_2026-04-01-09-29-26_3250320` | cell | vae_js (MSE) | 600 | 0.5 | start=300, over 100 | Config B gene list (10,134) |
| `run_2026-04-01-09-29-27_2189144` | cell | vae_nb | 600 | 0.5 | start=300, over 100 | Config B gene list (10,134) |
| job 3106245 (pending) | cell | vae_js (MSE) | 600 | 0.5 | start=300, over 100 | det>1% + exclude_mito (~24.7K) |

## Stage 2 Models

| Run Folder | Route | Decoder | Model | Gene filter | Loss schedule | Job |
|------------|-------|---------|-------|-------------|---------------|-----|
| `run_2026-03-13-00-55-11_1260842` | cell→nuclei | shared | vae_nb | det>1% (~24.7K) | old sequential | — |
| `run_2026-03-13-01-05-55_998747` | nuclei→cell | shared | vae_nb | det>1% (~24.7K) | old sequential | — |
| `run_2026-04-01-14-01-16_1946018` | joint | shared | vae_js | Config B (10,134) | cascading | 3073916 |
| `run_2026-04-01-14-43-14_475265` | joint | shared | vae_nb | Config B (10,134) | cascading | 3077244 |
| `run_2026-04-02-02-02-35_3345839` | joint | cond (embed=8) | vae_js | Config B (10,134) | cascading, xr=0 | 3102281 | **eval done**: cell=0.912, nuc→cell=0.756 |
| `run_2026-04-02-02-02-36_722169` | joint | cond (embed=8) | vae_nb | Config B (10,134) | cascading, xr=0 | 3102282 | **eval done**: cell=0.873, nuc→cell=0.688 |
| pending | joint | cond (embed=8) | vae_js | Config B (10,134) | cascading, xr=1.0 | 3106445 | running (~346/600) |
| pending (script ready) | joint | shared | vae_js | det>1% (~24.7K) | cascading | — |

---

### PerCT_5pct Runs (16,439 genes) — 2026-04-06+

**Gene list:** `data/gene_list_perct_5pct.csv` — 5% detection per cell type, 16,439 genes.

#### Stage 1 — cell only (vae_js, PerCT_5pct) — **completed**
- SLURM job: **3762768**
- Run folder: `run_2026-04-06-21-13-13_3762768`
- Script: `jobs/stage1/run_s1_cell_vae_js_pct5v2.sh`
- Model: vae_js (MSE), log1p, 600 epochs
- Architecture: internal_size=10,9, latent_size=5, embed_dim=8, 16,439 genes

#### Stage 2 — cond (no xr) — **completed**
- SLURM job: **3779214**
- Run folder: `run_2026-04-07-02-12-05_3779214`
- Stage 1 model: `run_2026-04-06-21-13-13_3762768/models/model.pt`
- Model: VAE_JSD_Cond, embed_dim=8, cascading annealing, xr=0
- Eval (job 3258060): gene corr 0.909, classif 88.7%, DE overlap 49.5%, var ratio 0.085

#### Stage 2 — cond + xr — **completed**
- SLURM job: **3551465**
- Run folder: `run_2026-04-07-02-12-05_3551465`
- Same as above but cross_recons=1.0 (annealed from epoch 100)
- Eval (job 3264971): pending

#### Stage 2 — cond + residual (SUC-6 Phase 3) — **submitted** (2026-04-07)
- SLURM job: **3265191**
- Script: `jobs/stage2/run_s2_cond_residual_pct5v2.sh`
- Stage 1 model: `run_2026-04-06-21-13-13_3762768/models/model.pt`
- Model: VAE_JSD_Cond, embed_dim=8, cascading annealing, xr=0
- **New:** residual_weight=5.0, annealed from epoch 100→200
- Residual MSE = MSE(decoded - CT_mean, target - CT_mean) — forces per-cell deviation reconstruction
- **Purpose:** SUC-6 Phase 3 — attack variance collapse by upweighting per-cell deviations in loss
- Loss schedule:
  - cell_recons=1.0, cell_jsd=0.1 (always on)
  - nuclei_recons=1.0, anneal 0→100
  - nuclei_jsd=0.1, delayed 100, anneal 100→200
  - alignment=1.0, delayed 200, anneal 200→300
  - residual=5.0, delayed 100, anneal 100→200

---

### Stage 1 Normalized Loss Probes (2026-05-01)

**Goal:** Make JSD actually affect the latent space in stage 1. Scale losses to [0,1] range.

| Job ID | Name | beta | Anneal | Notes |
|--------|------|------|--------|-------|
| 4153808 | s1_nl_b01 | 0.1 | None | JSD = 10% of recon (baseline) |
| 4153809 | s1_nl_b1 | 1.0 | None | Equal weight — strongest JSD signal |
| 4153810 | s1_nl_b1a | 1.0 | 0→1 over 50 epochs | Let recon settle first, then JSD ramps |
| 4153811 | lv_sparsity | — | — | Diagnostic: log_var vs sparsity (CPU) |

**Common params:** vae_js, Config B (10,134 genes), log1p, internal [1024,512], latent 32, lr=1e-4, batch 128, 20% subset, 200 epochs, `--normalize_losses True`, `--clamp_recons_loss False`

**Normalization method:**
- Recon: divide by epoch-0 average recon loss (auto-computed)
- JSD: divide by ln(2) = 0.693 (theoretical max)
- Both now ~[0,1], so beta directly controls balance

**Results (2026-05-08):**
- Winner: **4153808 (beta=0.1 no anneal)** — recon=0.251, JSD=0.08, clean UMAP clusters
- beta=1.0 over-regularizes: recon degrades to 0.312, JSD drops to 0.014 (latent collapses to N(0,1))
- mu/log_var healthy in all runs, no extremes

**Stage 2 follow-up:**

| Job ID | Config | Stage 1 |
|--------|--------|---------|
| 4514645 | cond decoder, WITH normalized losses | run_2026-05-01-12-01-55_1984590 |
| 4514646 | cond decoder, NO normalization | run_2026-05-01-12-01-55_1984590 |

Both: 600 epochs, same cascading anneal schedule as previous best cond runs

---

### Normloss-Fixed Systematic Sweep (2026-05-28)

**Context:** The `normalize_losses` bug was fixed on 2026-05-20 (stage 2 loops were ignoring the flag). This sweep retrains ALL conditioning variants from scratch with normloss actually working, providing a fair apples-to-apples comparison.

**Design:** 2x4 factorial — {log1p, CPM} x {learned_8d dec-only, scalar dec-only, scalar enc+dec (cVAE), learned_8d enc+dec (cVAE)}. All single-stage from scratch (no stage1 pretrain), 600 epochs, Config B (10,134 genes), normloss=True.

**Common params:** vae_js, internal [1024,512], latent 32, lr=1e-4, batch 128, 20% subset, beta=0.1, normalize_losses=True, alignment=1.0 (no explicit anneal in single-stage), xr=0.0, gene_list=training_gene_list_v3.csv

**Note:** "dec-only" = modality embedding in decoder only (cond_encoder=False). "cVAE" = modality embedding in both encoder and decoder (cond_encoder=True). "scalar" = binary 0/1 modality flag (embed_dim=1). "learned_8d" = learned 8-dim modality embedding.

#### Completed Runs (600 epochs)

| Norm | Conditioning | Run Folder | Notes |
|------|-------------|------------|-------|
| log1p | learned_8d dec-only | `run_2026-05-28-14-48-36_278184` | |
| log1p | scalar dec-only | `run_2026-05-28-15-15-37_31917` | |
| log1p | scalar cVAE (enc+dec) | `run_2026-05-28-15-15-37_1836209` | |
| log1p | learned_8d cVAE (enc+dec) | `run_2026-05-28-21-37-31_3423409` | 3 restarts (scavenge preemption) |
| CPM | learned_8d dec-only | `run_2026-05-28-15-15-38_1204566` | |
| CPM | scalar dec-only | `run_2026-05-28-15-23-50_1234766` | |
| CPM | scalar cVAE (enc+dec) | `run_2026-05-28-17-30-21_1880162` | |
| CPM | learned_8d cVAE (enc+dec) | `run_2026-05-29-03-10-28_3375023` | 4 restarts (scavenge preemption) |

#### Convergence Check Runs (400 epochs)

| Norm | Conditioning | Run Folder |
|------|-------------|------------|
| log1p | scalar dec-only | `run_2026-05-28-15-15-37_2686489` |
| log1p | scalar cVAE | `run_2026-05-28-15-15-38_1836210` |
| log1p | learned_8d cVAE | `run_2026-05-28-15-11-26_3334957` |
| CPM | learned_8d dec-only | `run_2026-05-28-15-15-38_1204567` |
| CPM | scalar dec-only | `run_2026-05-28-21-30-35_2745055` |
| CPM | scalar cVAE | `run_2026-05-28-17-29-33_1248184` |
| CPM | learned_8d cVAE | `run_2026-05-28-22-42-54_3359683` |

#### Inline eval metrics

Runs used `--stage3_skip_full_eval True`, so no inline gene corr / shift scores. Full evaluation pending via `analysis/eval/runner.py`.

#### Also: Normloss-fixed 2-stage retrain (2026-05-25)

- **log1p, learned_8d, dec-only, 2-stage** — job 5570485
  - Run: `run_2026-05-25-17-12-57_168455`
  - Stage 1: `run_2026-05-01-12-01-55_1984590` (normloss stage 1 winner, beta=0.1)
  - Same cascading anneal as original configB_cond, but normloss actually working
  - This is the only 2-stage run with normloss fixed; compare to from-scratch `_278184`

### cVAE seed replicates + split-aware benchmark (2026-10-09)

- **Seed replicates of the paper model** — job array **12646907** (`_1`, `_2`), script `jobs/stage2/run_single_stage_cvae_seed.sh`.
  Same config as job 7714771 / `run_2026-07-18-19-40-51_570984` plus `--seed 1|2` (new flag in `train.py`:
  seeds random/numpy/torch only; the 60/20/20 split stays at random_state=42) and `--stage3_save_epoch_dump False`
  (the original run's dumps are 166 GB; group quota is at 98%). scavenge_gpu, 400G, 24h, --requeue.
  **Status 2026-10-16: both done and benchmarked** (predict jobs 12686634/12686635, eval job 12686636, results in
  `run/benchmark/test/results/`, written up in `notes/benchmark_review_2026-10-09.md` §5a).
  - seed 1 = `run_2026-10-09-19-13-44_611868` (14 h 55, one attempt).
  - seed 2 = `run_2026-10-09-19-36-35_450293` is the folder the benchmark used. The array task was **preempted 9 times**
    and, lacking resume support, restarted from scratch into a new folder each time. Three attempts reached epoch 599
    (model.pt written after epoch 599; the two earlier ones were preempted during the final ~40 min eval/UMAP stage):
    attempt 2 `_450293` (used), attempt 9 `run_2026-10-10-22-29-59_3557416`, attempt 10 `run_2026-10-11-12-35-47_3752550`
    (the only one whose log survived; "Done" 2026-10-12 00:24). Same seed, different attempt (`_450293` vs `_3752550`,
    predict job 13442607 -> `run/benchmark/test/cvae_seed2_attempt10`): gene-mean r 0.9996, per-cell r 0.9987, but
    max |diff| 5.5 and within-CT per-gene variance log-corr 0.99 median / 0.84 worst type -> `--seed` fixes init and
    shuffling, not GPU nondeterminism; per-cell metrics carry run-to-run noise of this size even at fixed seed.
  - **Junk folders from the preempted attempts (~37 GB, group quota at 4087/4096 GiB on 2026-10-15), safe to delete:**
    `run_2026-10-09-19-14-14_3258901` (7.4G, 15 min), `run_2026-10-10-07-28-20_843449` (4.5G, epoch 99),
    `run_2026-10-10-10-08-33_3296376` (empty), `run_2026-10-10-10-19-16_3411188` (424M, epoch 449),
    `run_2026-10-10-18-54-00_3445873` (4.5G, no epochs), `run_2026-10-10-19-40-34_1227468` (13G, epoch 99),
    `run_2026-10-10-22-13-05_3473723` (6.8G, no epochs). `_3557416` (540M, complete) is redundant but harmless.
- **Library-size covariate retrain (arms A/B)** — job array **13445566** (`_1` = arm A, `_2` = arm B), submitted 2026-10-16
  on `pi_gerstein_gpu,scavenge_gpu` (group-owned non-preemptible GPU partition: r813u29n03 1 TB A100, r109u24n01/r109u32n01
  rtx3090 167 GB; both tasks started on r813u29n03 within the hour). First submission 13442818 on scavenge_gpu alone had
  an estimated start of 2026-10-20 and was cancelled. Peak RSS of this config is 326 GiB (sacct 12646907), so 400G is needed.
  **Broken GPU:** `13445566_1` (arm A) was allocated GPU `0000:31:00.0` on r813u29n03, which nvidia-smi reports as
  "Unable to determine the device handle ... Unknown Error" (that is why it was the free GPU). `imputeVAE/utils.py` falls
  back to CPU silently, so the task trained epoch 0 on CPU (~9 min/epoch) before being cancelled; its folder was deleted.
  Arm A resubmitted as **13446475** (`--array=1 --gpus=2`): the script now picks the first healthy GPU by UUID
  (`CUDA_VISIBLE_DEVICES=GPU-...`) and exits 3 instead of training on CPU. Arm B = `13445566_2` (healthy A100 CA:00.0).
  Report the broken GPU to YCRC (hpc@yale.edu) so it gets drained.
  script `jobs/stage2/run_single_stage_cvae_libsize.sh`, plan `notes/libsize_covariate_review_2026-10-09.md` §3/§5.
  Seed-1 paper config + `--libsize_cond True --libsize_target ct_median_cell`; arm A keeps per-modality JSD 0.1/0.1,
  arm B lowers them to 0.01/0.01 (alignment 1.0 in both). Fixed run folders via the new `--run_name`:
  `run/run_2026-10-16_libsize_armA_jsd0.1_seed1`, `run/run_2026-10-16_libsize_armB_jsd0.01_seed1`; a requeued task
  resumes from `models/model.pt` + `run_state.json` (every 10 epochs) instead of restarting. Compare arm A against seed 1
  (`_611868`, same seed, no covariate). Evaluate: predict with `CVAE_TARGET_LIBSIZE=ct_median_cell` (and `quantile_map`,
  `source` as ablations) -> `run/benchmark/test/cvae_libsize_arm{A,B}[_<target>]`, then `evaluate_all.sh`; then
  `analysis/latent_usage_check.py` and `analysis/variance_zero_nonzero_fullmodel.py` (parametrised 2026-10-16, see the review note §5).
  **Status 2026-10-23: both arms trained to completion on r813u29n03 A100s** (arm A 13446475_1, 12 h 06, final loss 0.958,
  nuclei/cell JSD 0.023/0.024; arm B 13445566_2, 12 h 13, final loss 0.959, JSD 0.039/0.041; both `run_state.json` epoch 599,
  `libsize_stats.json` written; final inline valid cross-decode gene corr 0.989/0.987). Evaluation submitted 2026-10-23:
  predict jobs 13779998-13780003 (A/B x ct_median_cell, quantile_map, source -> `run/benchmark/test/cvae_libsize_arm{A,B}[_qmap|_source]`),
  `evaluate_all` 13780004, diagnostics array 13780170 (`jobs/eval/libsize_arm_diagnostics.sh`: export -> latent usage -> variance,
  outputs `run/libsize_arm{A,B}_eval/`, `notes/experiments/libsize_covariate/arm{A,B}/`). Results: `notes/libsize_covariate_review_2026-10-09.md` §6.
  **Results (2026-10-23, review note §6):** library-size share of imputed within-type variance 85 % -> 0 %; within-type
  variance falls 11x and sits on 2-3 identity axes (PR 1.6); arm B (JSD 0.01) no different -> collapse is objective-driven,
  arm C dropped. Cross-chemistry (v2, job 13783381): arm A at the v2-cell library gene corr 0.93 / R² 0.72 vs paper 0.59 / -0.03.
- **Latent-width control** — job array **13784373** (`jobs/stage2/run_single_stage_cvae_libsize_latent.sh`), submitted 2026-10-23
  on `pi_gerstein_gpu,scavenge_gpu` with `--gpus=2` + GPU guard: arm A config (covariate, JSD 0.1/0.1, seed 1) with
  `--latent_size 3` (8 dims, `_1`) and `2` (4 dims, `_2`); folders `run/run_2026-10-23_libsize_armA_latent{8,4}_seed1`.
  Purpose: ablation showing capacity is not the limit (32-dim arms use 4-5 dims). Evaluate like the arms (predict_cvae +
  evaluate_all + latent_usage_check).
  **Status 2026-10-23 12:10: both COMPLETED** on r813u29n03 (8 dims 11 h 25, final loss 0.950, JSD 0.066/0.067;
  4 dims 11 h 15, final loss 0.946, JSD 0.075/0.076; 32-dim arm A was 0.958). Eval submitted 2026-10-23 13:2x: benchmark
  decodes 13822181-13822188 (`run/benchmark/test/cvae_libsize_latent{8,4}[_qmap|_source|_gqmap]`), full table 13822189
  then 13822272, diagnostics array 13822273 (tasks 3/4 of `jobs/eval/libsize_arm_diagnostics.sh`), add-on 13823088,
  quick eval 13823078 -> `run/benchmark/test/results_latent_width/`. **Results (review note §6.7):** 8 dims -> 2 active
  dims, gene corr 0.957 / R² 0.90 / clf 0.935 (L5 ET + Sncg merged); 4 dims -> 1 active dim, 0.910 / 0.80 / clf 0.56;
  within-type residual 61 -> 9 -> 3 and its gene-gene structure 0.48 -> 0.19 -> 0.02. Capacity is not the limit; 32 dims kept.
- **Benchmark of the covariate arms (13780004) DONE 2026-10-23**: with the per-type cell library (label) arm A gene corr
  0.970 / R² 0.927, quantile-map decode energy 2.6 (best in table); label-free decodes 0.84-0.87. Label-free scale modes
  added 2026-10-23 (`global_shift`, `global_quantile_map`, `knn_cell_median`, `knn_quantile_map` in
  `analysis/benchmark/predict_cvae.py`; `global_*_target`, `global_shift_train`, `knn_*_target` in
  `analysis/cross_dataset_eval.py`; `knn_type_*` = type predicted from the 50 nearest training nuclei): predicts
  13819587/8, 13822255/6, 13822363/4; quick evals 13819596, 13822258, 13822366 -> `run/benchmark/test/results_libfree_scale/`;
  v2 arrays 13819589_5-7, 13822257_8-10, 13822365_11-12; final full table 13822272. **Result:** predicted-type quantile map
  0.962 / R² 0.91 / energy 2.6 with no inference label (labelled 0.970; no labels anywhere 0.921; paper model 0.903);
  on v2 0.888 (type prediction 77 % across chemistry). Record: review note §6.4 / §6.4a.
- **v2 re-run with the reference classifier (14189131_13-14, 2026-10-27, 20 min CPU each)**: `cross_dataset_eval.py
  --type_predictor pca_lr` (new flag; `jobs/eval/run_cross_dataset_v2_libsize.sh` array 13-14) ->
  `run/cross_dataset_eval/v2_batch_armA_lrtype_{v2,train_shift}`. Type accuracy on v2 0.77 -> 0.84, label-free gene corr
  0.888 -> 0.903 (label 0.926), train-shift mode 0.814 -> 0.833. Residual: Pvalb -> Sst 56 %, L6 IT <-> L5 IT split.
- **Split-aware benchmark redo** — see `notes/benchmark_review_2026-10-09.md`; jobs listed in `jobs/benchmark/submitted_jobs.txt`;
  results in `run/benchmark/test/results/` (and `run/benchmark/valid/results/`).
- **Ablations on the final covariate model** — job array **14861675** (`jobs/ablation/run_ablations_final_model.sh`),
  submitted 2026-11-04 on `pi_gerstein_gpu,scavenge_gpu` (`--gpus=2` + GPU guard, 400G, 24 h, requeue/resume). Arm A
  config (covariate, JSD 0.1/0.1, alignment 1.0, seed 1, 600 ep) with one change per task: `_1` nojsd (JSD 0/0),
  `_2` noalign (alignment 0), `_3` deccond (`--cond_encoder False`), `_4` n1_L5IT (`--exclude_cell_types "L5 IT"`),
  `_5` n1_Pvalb, `_6` donor5 (`--holdout_donors nuclei:5`). New flags apply after the fixed split (train/valid only;
  test = manifest). Folders `run/run_2026-11-04_ablation_<arm>_seed1`. Eval chain: predict array 14861676 (aftercorr;
  `run/benchmark/test/abl_<arm>_{ctmed,qmap,ktqmap,kqmap}`), evaluate 14861677 (afterok; `results_ablations/`,
  `results_ablations/by_donor/` via new `analysis/benchmark/eval_by_donor.py`). Record:
  `notes/experiments/ablations_final_model/README.md`.
  2026-11-05: all six arms finished (10-15 h; tasks 3/4/5 preempted and resumed). deccond was preempted at the instant
  of exit -> sacct PREEMPTED -> predict `_3` never ran, evaluate 14861677 cancelled. Rerun: predict 15559184_3 (deccond),
  15559684_[4-6] (n1_L5IT, n1_Pvalb, donor5, after the `predict_cvae` kNN-reference leak fix: the label-free decodes
  had used the cache's unfiltered train partition, so the type vote named the held-out type), evaluate 15559685.
  Debug logs `slurm-14861675_{1,2,3,6}_cvae_ablation.out` (9.4 GB) to delete by hand.
  Results 2026-11-06 (`results_ablations/summary.md`, README Results section): nojsd 0.964 / noalign 0.947 (clf 0.917) /
  deccond 0.963 vs arm A 0.970; N-1 L5 IT delta r 0.93, Pvalb 0.85-0.87 with identity collapse (recall ~0); donor 5
  hold-out 0 cost labelled, -0.01 label-free.
- **Library (batch) covariate** — 2026-11-06, `jobs/ablation/run_library_cond.sh`, run `run/run_2026-11-06_libcond_seed1`:
  arm A config + `--library_cond True --library_dim 4 --library_capacity 32 --library_dropout 0.1` (learned embedding
  of meta_batch_id on encoder + decoder, id 0 = UNKNOWN / marginal library; `library_vocab.json`). Chain: smoke 15562731
  (CPU, 10 %, 2 epochs) -> train 15562732 -> predict 15562733 (`run/benchmark/test/libcond_*`, incl. female/male
  target-library decodes) -> evaluate 15563057 (`results_libcond/`). Record: ablations README, library covariate section.
  Results (read 2026-11-13): labelled decode 0.956 / R² 0.893 (arm A 0.970 / 0.927), label-free 0.959 / 0.899 (0.962 /
  0.909), clf 0.927 (0.967); rare cell types lose identity (L5 ET recall 0.00, filed as L6 CT; Endo, Oligo, Macrophage
  profile corr -0.05 to -0.12); cross-modality latent mixing worse (purity 0.78 vs 0.64). Positive: female / male
  target-library decodes reproduce the real sex difference (Xist 3.6 vs 0.0; genome-wide r 0.655). Not the reported
  model; supplementary (sex-placement panel).
