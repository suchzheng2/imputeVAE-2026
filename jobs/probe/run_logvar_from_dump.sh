#!/bin/bash
#SBATCH --job-name=lv_dump
#SBATCH --partition=pi_gerstein
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --output=jobs/logs/slurm-%j_logvar_from_dump.out
#SBATCH --error=jobs/logs/slurm-%j_logvar_from_dump.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

python3 << 'PYEOF'
import json
import numpy as np

run = 'run/run_2026-04-01-09-29-26_3250320'
dump_path = f'{run}/epoch_dump/train_epoch_599.json'

print("Loading epoch dump (this may take a minute)...")
with open(dump_path) as f:
    data = json.load(f)

all_logvar = []
all_mu = []
all_celltypes = []

for celltype, batches in data.items():
    for batch_id, batch_data in batches.items():
        lv = np.array(batch_data['log_var'], dtype=np.float32)
        mu_arr = np.array(batch_data['mu'], dtype=np.float32)
        all_logvar.append(lv)
        all_mu.append(mu_arr)
        ct_labels = batch_data.get('cell_type_original', [celltype]*lv.shape[0])
        all_celltypes.extend(ct_labels)

del data  # free memory

logvar = np.concatenate(all_logvar, axis=0)
mu = np.concatenate(all_mu, axis=0)
del all_logvar, all_mu

print(f"Total cells: {logvar.shape[0]}, Latent dims: {logvar.shape[1]}")
print(f"\n=== log_var stats ===")
print(f"  min: {logvar.min():.3f}, max: {logvar.max():.3f}")
print(f"  mean: {logvar.mean():.3f}, std: {logvar.std():.3f}")
print(f"\n=== mu stats ===")
print(f"  min: {mu.min():.3f}, max: {mu.max():.3f}")
print(f"  mean: {mu.mean():.3f}, std: {mu.std():.3f}")

# Per-dimension stats
print(f"\n=== Per-latent-dimension log_var ===")
print(f"{'dim':>4s} {'min':>8s} {'max':>8s} {'mean':>8s} {'std':>8s}")
for d in range(logvar.shape[1]):
    print(f"{d:4d} {logvar[:,d].min():8.2f} {logvar[:,d].max():8.2f} {logvar[:,d].mean():8.2f} {logvar[:,d].std():8.2f}")

# How many hit the clamp?
n_at_neg20 = np.sum(logvar < -19.9)
n_at_pos20 = np.sum(logvar > 19.9)
print(f"\nElements near -20 clamp: {n_at_neg20} / {logvar.size} ({100*n_at_neg20/logvar.size:.2f}%)")
print(f"Elements near +20 clamp: {n_at_pos20} / {logvar.size} ({100*n_at_pos20/logvar.size:.2f}%)")

# Per-cell: mean log_var vs cell type
celltypes_arr = np.array(all_celltypes)
unique_cts = np.unique(celltypes_arr)
print(f"\n=== Per-cell-type mean log_var & mu ===")
print(f"{'cell_type':>20s} {'n':>6s} {'mean_lv':>8s} {'std_lv':>8s} {'mean_|mu|':>10s} {'mu_range':>10s}")
for ct in sorted(unique_cts):
    mask = celltypes_arr == ct
    ct_lv = logvar[mask]
    ct_mu = mu[mask]
    print(f"{ct:>20s} {mask.sum():6d} {ct_lv.mean():8.3f} {ct_lv.std():8.3f} {np.abs(ct_mu).mean():10.3f} {ct_mu.max()-ct_mu.min():10.3f}")

# Histogram
bins = np.linspace(-25, 25, 51)
counts, edges = np.histogram(logvar.flatten(), bins=bins)
print(f"\n=== log_var histogram ===")
for i in range(len(counts)):
    if counts[i] > 0:
        bar = '#' * min(60, int(60 * counts[i] / counts.max()))
        print(f"  [{edges[i]:6.1f}, {edges[i+1]:6.1f}): {counts[i]:8d} {bar}")

print("\nDone.")
PYEOF
