#!/usr/bin/env python3
"""
10_global_imputation_qc.py
Global QC and visualization for Human M1 zero-shot transfer.

Analyses:
  A. PCA/UMAP global embedding (raw nucleus vs imputed cell-like)
  B. Cell-type centroid arrows
  C. Cell identity preservation by nearest centroid
  D. Program-score shifts (SynGO, TF, ribosomal, nuclear lncRNA)
  E. Distributional delta summary
  F. Optional mouse-human projected visualization
"""
import os, sys, csv, glob, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ──
BASE = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
HM1 = os.path.join(BASE, "validation/human_m1_transfer")
FIG_DIR = os.path.join(HM1, "figures")
RES_DIR = os.path.join(HM1, "results")
REP_DIR = os.path.join(HM1, "reports")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

NUC_PATH = os.path.join(HM1, "processed/human_m1_nuclei_mouse_model_order.npz")
IMP_PATH = os.path.join(HM1, "inference/human_m1_imputed_cell_log1p.npy")
OBS_PATH = os.path.join(HM1, "inference/human_m1_inference_obs.csv")
GENE_MAP_PATH = os.path.join(HM1, "inference/human_m1_inference_gene_map.csv")
MASK_PATH = os.path.join(HM1, "processed/human_m1_available_gene_mask.npy")

SYNGO_DELTA_PATH = os.path.join(BASE, "validation/syngo/syngo_brain_validation/syngo_gene_celltype_deltas.csv")
TF_DELTA_PATH = os.path.join(BASE, "validation/tf/tf_brain_validation/tf_gene_celltype_deltas.csv")
MOUSE_DELTA_DIR = os.path.join(BASE, "notes/experiments/delta_analysis/tables")

SEED = 42
MAX_PER_CLASS = 500
N_HVG = 2000
N_PCA = 50

# ── helpers ──
def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png/pdf")

# ── load data ──
print("=== Loading data ===")
nuc_sparse = sp.load_npz(NUC_PATH)
imputed = np.load(IMP_PATH, mmap_mode="r")
obs = pd.read_csv(OBS_PATH)
gene_map = pd.read_csv(GENE_MAP_PATH)
mask = np.load(MASK_PATH)  # boolean, True = mapped

mapped_idx = np.where(mask)[0]
n_mapped = len(mapped_idx)
mouse_genes = gene_map["mouse_model_gene"].values
mapped_genes_set = set(mouse_genes[mapped_idx])

print(f"  Nuclei: {nuc_sparse.shape[0]}, Genes: {nuc_sparse.shape[1]}, Mapped: {n_mapped}")

# ── build gene name -> mapped column index lookup ──
gene_to_mapped_col = {}
for i, gi in enumerate(mapped_idx):
    gene_to_mapped_col[mouse_genes[gi]] = i

# ──────────────────────────────────────────────────────────────────────
# A. PCA/UMAP global embedding
# ──────────────────────────────────────────────────────────────────────
print("\n=== Part A: PCA/UMAP global embedding ===")

# balanced subsample
rng = np.random.RandomState(SEED)
classes = obs["mapped_mouse_like_class"].values
unique_classes = sorted(set(classes))
sampled_idx = []
for cls in unique_classes:
    cls_idx = np.where(classes == cls)[0]
    if len(cls_idx) > MAX_PER_CLASS:
        chosen = rng.choice(cls_idx, MAX_PER_CLASS, replace=False)
    else:
        chosen = cls_idx
    sampled_idx.append(chosen)
sampled_idx = np.sort(np.concatenate(sampled_idx))
n_sampled = len(sampled_idx)
print(f"  Sampled {n_sampled} nuclei from {len(unique_classes)} classes (max {MAX_PER_CLASS}/class)")

# extract mapped-gene matrices for sampled nuclei
nuc_sampled = nuc_sparse[sampled_idx][:, mapped_idx]
if sp.issparse(nuc_sampled):
    nuc_sampled = nuc_sampled.toarray()
imp_sampled = imputed[sampled_idx][:, mapped_idx]

# HVG selection by variance in raw nucleus data
gene_var = np.var(nuc_sampled, axis=0)
hvg_order = np.argsort(gene_var)[::-1]
n_hvg = min(N_HVG, n_mapped)
hvg_idx = np.sort(hvg_order[:n_hvg])
print(f"  HVG: top {n_hvg} by variance (of {n_mapped} mapped)")

nuc_hvg = nuc_sampled[:, hvg_idx]
imp_hvg = imp_sampled[:, hvg_idx]

# stack: [nucleus rows; imputed rows]
combined = np.vstack([nuc_hvg, imp_hvg])
modality_labels = np.array(["nucleus"] * n_sampled + ["imputed"] * n_sampled)
broad_labels = np.concatenate([
    obs["mapped_broad_class"].values[sampled_idx],
    obs["mapped_broad_class"].values[sampled_idx],
])
mouse_like_labels = np.concatenate([
    obs["mapped_mouse_like_class"].values[sampled_idx],
    obs["mapped_mouse_like_class"].values[sampled_idx],
])

# standardize and PCA
scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined)
pca = PCA(n_components=N_PCA, random_state=SEED)
pca_coords = pca.fit_transform(combined_scaled)
print(f"  PCA: {N_PCA} components, explained variance {pca.explained_variance_ratio_.sum():.3f}")

# attempt UMAP
umap_coords = None
try:
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=SEED)
    umap_coords = reducer.fit_transform(pca_coords)
    print("  UMAP: computed successfully")
    umap_available = True
except ImportError:
    print("  UMAP: not available, skipping")
    umap_available = False

# save embedding CSV
emb_rows = []
for i in range(len(modality_labels)):
    row = {
        "sample_index": sampled_idx[i % n_sampled],
        "modality": modality_labels[i],
        "mapped_broad_class": broad_labels[i],
        "mapped_mouse_like_class": mouse_like_labels[i],
        "PC1": pca_coords[i, 0],
        "PC2": pca_coords[i, 1],
    }
    if umap_coords is not None:
        row["UMAP1"] = umap_coords[i, 0]
        row["UMAP2"] = umap_coords[i, 1]
    emb_rows.append(row)
emb_df = pd.DataFrame(emb_rows)
emb_df.to_csv(os.path.join(RES_DIR, "human_m1_global_embedding.csv"), index=False)
print(f"  Saved embedding CSV ({len(emb_df)} rows)")

# ── plot A ──
broad_classes_sorted = sorted(set(broad_labels))
broad_cmap = plt.cm.get_cmap("tab10", len(broad_classes_sorted))
broad_color_map = {c: broad_cmap(i) for i, c in enumerate(broad_classes_sorted)}

n_panels = 4 if umap_available else 2
fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
if n_panels == 2:
    axes = list(axes)

# Panel 1: PCA by modality
ax = axes[0]
for mod, color, marker in [("nucleus", "#4e79a7", "o"), ("imputed", "#e15759", "^")]:
    sel = modality_labels == mod
    ax.scatter(pca_coords[sel, 0], pca_coords[sel, 1], c=color, s=2, alpha=0.3,
               label=mod, marker=marker, rasterized=True)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("PCA: modality")
ax.legend(markerscale=5, fontsize=8)

# Panel 2: PCA by broad class
ax = axes[1]
for cls in broad_classes_sorted:
    sel = broad_labels == cls
    ax.scatter(pca_coords[sel, 0], pca_coords[sel, 1], c=[broad_color_map[cls]],
               s=2, alpha=0.3, label=cls, rasterized=True)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("PCA: broad class")
ax.legend(markerscale=5, fontsize=7, loc="best", ncol=1)

if umap_available:
    # Panel 3: UMAP by modality
    ax = axes[2]
    for mod, color, marker in [("nucleus", "#4e79a7", "o"), ("imputed", "#e15759", "^")]:
        sel = modality_labels == mod
        ax.scatter(umap_coords[sel, 0], umap_coords[sel, 1], c=color, s=2, alpha=0.3,
                   label=mod, marker=marker, rasterized=True)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("UMAP: modality")
    ax.legend(markerscale=5, fontsize=8)

    # Panel 4: UMAP by broad class
    ax = axes[3]
    for cls in broad_classes_sorted:
        sel = broad_labels == cls
        ax.scatter(umap_coords[sel, 0], umap_coords[sel, 1], c=[broad_color_map[cls]],
                   s=2, alpha=0.3, label=cls, rasterized=True)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("UMAP: broad class")
    ax.legend(markerscale=5, fontsize=7, loc="best", ncol=1)

fig.suptitle(f"Human M1 Global Embedding: {n_sampled} nuclei, {n_hvg} HVGs", fontsize=12, y=1.02)
fig.tight_layout()
save_fig(fig, "human_m1_global_pca_umap")


# ──────────────────────────────────────────────────────────────────────
# B. Cell-type centroid arrows
# ──────────────────────────────────────────────────────────────────────
print("\n=== Part B: Centroid arrows ===")

nuc_pca = pca_coords[:n_sampled]
imp_pca = pca_coords[n_sampled:]

# also need per-cell predicted delta for summary
nuc_all_mapped = nuc_sampled  # (n_sampled, n_mapped)
imp_all_mapped = imp_sampled
delta_all_mapped = imp_all_mapped - nuc_all_mapped  # per-cell, per-gene

centroid_rows = []
for level_name, level_labels in [("broad_class", obs["mapped_broad_class"].values[sampled_idx]),
                                  ("mouse_like_class", obs["mapped_mouse_like_class"].values[sampled_idx])]:
    for cls in sorted(set(level_labels)):
        sel = level_labels == cls
        n_sel = sel.sum()
        nuc_cent = nuc_pca[sel].mean(axis=0)
        imp_cent = imp_pca[sel].mean(axis=0)
        shift_dist = np.sqrt(np.sum((imp_cent - nuc_cent) ** 2))
        cls_delta = delta_all_mapped[sel].mean(axis=1)  # mean over genes per cell
        centroid_rows.append({
            "class_level": level_name,
            "class": cls,
            "n_nuclei": n_sel,
            "pc1_nucleus": nuc_cent[0],
            "pc2_nucleus": nuc_cent[1],
            "pc1_imputed": imp_cent[0],
            "pc2_imputed": imp_cent[1],
            "centroid_shift_distance": shift_dist,
            "mean_predicted_delta_all_mapped": cls_delta.mean(),
            "median_predicted_delta_all_mapped": np.median(cls_delta),
        })

centroid_df = pd.DataFrame(centroid_rows)
centroid_df.to_csv(os.path.join(RES_DIR, "human_m1_centroid_shift_summary.csv"), index=False)
print(f"  Saved centroid shift summary ({len(centroid_df)} rows)")

# plot arrows
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_i, (level_name, title) in enumerate([("broad_class", "Broad Class"),
                                              ("mouse_like_class", "Mouse-like Class")]):
    ax = axes[ax_i]
    sub = centroid_df[centroid_df["class_level"] == level_name]

    # background scatter (faint)
    level_col = "mapped_broad_class" if level_name == "broad_class" else "mapped_mouse_like_class"
    bg_labels = obs[level_col].values[sampled_idx]
    ax.scatter(nuc_pca[:, 0], nuc_pca[:, 1], c="lightgray", s=1, alpha=0.1, rasterized=True)

    for _, row in sub.iterrows():
        color = "red" if row["n_nuclei"] < 200 else "black"
        lw = 1.0 if row["n_nuclei"] < 200 else 1.5
        ax.annotate("", xy=(row["pc1_imputed"], row["pc2_imputed"]),
                     xytext=(row["pc1_nucleus"], row["pc2_nucleus"]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=lw))
        ax.text(row["pc1_nucleus"], row["pc2_nucleus"], row["class"],
                fontsize=6, ha="center", va="bottom", color=color)

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Centroid Arrows: {title}")
    # legend for low-sample
    ax.plot([], [], "->", color="red", label="n < 200 nuclei")
    ax.plot([], [], "->", color="black", label="n >= 200 nuclei")
    ax.legend(fontsize=8)

fig.tight_layout()
save_fig(fig, "human_m1_centroid_arrows")


# ──────────────────────────────────────────────────────────────────────
# C. Cell identity preservation
# ──────────────────────────────────────────────────────────────────────
print("\n=== Part C: Cell identity preservation ===")

identity_rows = []
for level_name, level_labels_all in [("broad_class", obs["mapped_broad_class"].values),
                                       ("mouse_like_class", obs["mapped_mouse_like_class"].values)]:
    level_labels = level_labels_all[sampled_idx]
    unique_cls = sorted(set(level_labels))

    # compute raw nucleus centroids per class
    centroids = {}
    for cls in unique_cls:
        sel = level_labels == cls
        centroids[cls] = nuc_pca[sel].mean(axis=0)
    centroid_matrix = np.array([centroids[cls] for cls in unique_cls])  # (n_cls, n_pca)

    # assign each raw nucleus to nearest centroid
    raw_dists = cdist(nuc_pca, centroid_matrix)
    raw_assignments = np.array([unique_cls[j] for j in raw_dists.argmin(axis=1)])

    # assign each imputed profile to nearest raw-nucleus centroid
    imp_dists = cdist(imp_pca, centroid_matrix)
    imp_assignments = np.array([unique_cls[j] for j in imp_dists.argmin(axis=1)])

    for cls in unique_cls:
        sel = level_labels == cls
        n_sel = sel.sum()
        raw_self = (raw_assignments[sel] == cls).mean()
        imp_self = (imp_assignments[sel] == cls).mean()
        identity_rows.append({
            "class_level": level_name,
            "class": cls,
            "n_sampled": n_sel,
            "raw_self_assignment_rate": raw_self,
            "imputed_assignment_to_original_class_rate": imp_self,
            "drop_from_raw": raw_self - imp_self,
        })

identity_df = pd.DataFrame(identity_rows)
identity_df.to_csv(os.path.join(RES_DIR, "human_m1_identity_preservation.csv"), index=False)
print(f"  Saved identity preservation ({len(identity_df)} rows)")

# plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax_i, level_name in enumerate(["broad_class", "mouse_like_class"]):
    ax = axes[ax_i]
    sub = identity_df[identity_df["class_level"] == level_name].sort_values("imputed_assignment_to_original_class_rate")
    colors = ["red" if r["n_sampled"] < 200 else "#4e79a7" for _, r in sub.iterrows()]
    ax.barh(range(len(sub)), sub["imputed_assignment_to_original_class_rate"].values, color=colors)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["class"].values, fontsize=7)
    ax.set_xlabel("Imputed → original class rate")
    ax.set_title(f"Identity Preservation ({level_name})")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    # legend
    ax.barh([], [], color="red", label="n < 200")
    ax.barh([], [], color="#4e79a7", label="n >= 200")
    ax.legend(fontsize=8)

fig.tight_layout()
save_fig(fig, "human_m1_identity_preservation")


# ──────────────────────────────────────────────────────────────────────
# D. Program-score shifts
# ──────────────────────────────────────────────────────────────────────
print("\n=== Part D: Program-score shifts ===")

# load gene sets
def load_gene_set_from_deltas(path):
    genes = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            genes.add(row["gene"])
    return genes

syngo_genes = load_gene_set_from_deltas(SYNGO_DELTA_PATH) & mapped_genes_set
tf_genes = load_gene_set_from_deltas(TF_DELTA_PATH) & mapped_genes_set
ribo_genes = set(g for g in mapped_genes_set if g.startswith("Rpl") or g.startswith("Rps"))
nuclear_lnc_candidates = {"Malat1", "Neat1", "Meg3", "Xist"}
nuclear_lnc_genes = nuclear_lnc_candidates & mapped_genes_set

gene_sets = {
    "SynGO": syngo_genes,
    "TF": tf_genes,
    "Ribosomal": ribo_genes,
    "Nuclear_lncRNA": nuclear_lnc_genes,
}
for gs_name, gs in gene_sets.items():
    print(f"  {gs_name}: {len(gs)} mapped genes")

# compute scores on FULL data (not just sampled)
# Use memory-mapped imputed + sparse nucleus, iterate by broad class and mouse-like class
print("  Computing program scores (all nuclei)...")

program_rows = []
for level_name, level_col in [("broad_class", "mapped_broad_class"),
                                ("mouse_like_class", "mapped_mouse_like_class")]:
    level_labels = obs[level_col].values
    for cls in sorted(set(level_labels)):
        cls_idx = np.where(level_labels == cls)[0]
        n_cls = len(cls_idx)

        # extract nucleus and imputed for this class, mapped genes only
        nuc_cls = nuc_sparse[cls_idx][:, mapped_idx]
        if sp.issparse(nuc_cls):
            nuc_cls = nuc_cls.toarray()
        imp_cls = imputed[cls_idx][:, mapped_idx]

        for gs_name, gs in gene_sets.items():
            if len(gs) == 0:
                continue
            # find column indices within mapped-gene submatrix
            gs_cols = [gene_to_mapped_col[g] for g in gs if g in gene_to_mapped_col]
            if len(gs_cols) == 0:
                continue
            gs_cols = np.array(gs_cols)

            nuc_scores = nuc_cls[:, gs_cols].mean(axis=1)
            imp_scores = imp_cls[:, gs_cols].mean(axis=1)
            delta_scores = imp_scores - nuc_scores

            program_rows.append({
                "class_level": level_name,
                "class": cls,
                "gene_set": gs_name,
                "n_genes_used": len(gs_cols),
                "n_nuclei": n_cls,
                "nucleus_score_mean": float(nuc_scores.mean()),
                "imputed_score_mean": float(imp_scores.mean()),
                "score_delta_mean": float(delta_scores.mean()),
                "score_delta_median": float(np.median(delta_scores)),
            })

program_df = pd.DataFrame(program_rows)
program_df.to_csv(os.path.join(RES_DIR, "human_m1_program_score_shifts.csv"), index=False)
print(f"  Saved program score shifts ({len(program_df)} rows)")

# plot: grouped barplot by broad class
fig, ax = plt.subplots(figsize=(12, 6))
broad_sub = program_df[program_df["class_level"] == "broad_class"].copy()
gs_names = sorted(broad_sub["gene_set"].unique())
broad_cls_order = sorted(broad_sub["class"].unique())
n_gs = len(gs_names)
bar_width = 0.8 / n_gs
x = np.arange(len(broad_cls_order))

gs_colors = {"SynGO": "#e15759", "TF": "#4e79a7", "Ribosomal": "#59a14f", "Nuclear_lncRNA": "#b07aa1"}
for i, gs in enumerate(gs_names):
    vals = []
    for cls in broad_cls_order:
        match = broad_sub[(broad_sub["class"] == cls) & (broad_sub["gene_set"] == gs)]
        vals.append(match["score_delta_mean"].values[0] if len(match) > 0 else 0)
    ax.bar(x + i * bar_width, vals, bar_width, label=gs, color=gs_colors.get(gs, "gray"))

ax.set_xticks(x + bar_width * (n_gs - 1) / 2)
ax.set_xticklabels(broad_cls_order, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mean score delta (imputed - nucleus)")
ax.set_title("Program Score Shifts by Broad Class")
ax.legend(fontsize=8)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
fig.tight_layout()
save_fig(fig, "human_m1_program_score_shifts")


# ──────────────────────────────────────────────────────────────────────
# E. Distributional delta summary
# ──────────────────────────────────────────────────────────────────────
print("\n=== Part E: Distributional delta summary ===")

dist_rows = []
for level_name, level_col in [("broad_class", "mapped_broad_class"),
                                ("mouse_like_class", "mapped_mouse_like_class")]:
    level_labels = obs[level_col].values
    for cls in sorted(set(level_labels)):
        cls_idx = np.where(level_labels == cls)[0]
        # pseudobulk: mean across cells per gene (mapped genes)
        nuc_mean = nuc_sparse[cls_idx][:, mapped_idx].mean(axis=0)
        if sp.issparse(nuc_mean) or hasattr(nuc_mean, 'A1'):
            nuc_mean = np.asarray(nuc_mean).flatten()
        imp_mean = imputed[cls_idx][:, mapped_idx].mean(axis=0)
        gene_delta = imp_mean - nuc_mean

        dist_rows.append({
            "class_level": level_name,
            "class": cls,
            "n_nuclei": len(cls_idx),
            "n_genes": len(gene_delta),
            "mean_delta": float(gene_delta.mean()),
            "median_delta": float(np.median(gene_delta)),
            "fraction_positive_delta": float((gene_delta > 0).mean()),
            "delta_iqr": float(np.percentile(gene_delta, 75) - np.percentile(gene_delta, 25)),
            "fraction_large_positive_delta": float((gene_delta > 0.5).mean()),
            "fraction_negative_delta": float((gene_delta < 0).mean()),
        })

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(os.path.join(RES_DIR, "human_m1_distributional_delta_summary.csv"), index=False)
print(f"  Saved distributional delta summary ({len(dist_df)} rows)")

# violin plot for broad classes
broad_dist = dist_df[dist_df["class_level"] == "broad_class"].copy()
broad_cls_order_e = sorted(broad_dist["class"].unique())

# collect gene-level deltas for violin
print("  Computing gene-level deltas for violin plot...")
violin_data = []
violin_labels = []
for cls in broad_cls_order_e:
    cls_idx = np.where(obs["mapped_broad_class"].values == cls)[0]
    nuc_mean = nuc_sparse[cls_idx][:, mapped_idx].mean(axis=0)
    if hasattr(nuc_mean, 'A1'):
        nuc_mean = np.asarray(nuc_mean).flatten()
    imp_mean = imputed[cls_idx][:, mapped_idx].mean(axis=0)
    gene_delta = imp_mean - nuc_mean
    violin_data.append(gene_delta)
    violin_labels.append(cls)

fig, ax = plt.subplots(figsize=(12, 6))
parts = ax.violinplot(violin_data, positions=range(len(violin_labels)), showmeans=True, showmedians=True)
for pc in parts["bodies"]:
    pc.set_alpha(0.6)
ax.set_xticks(range(len(violin_labels)))
ax.set_xticklabels(violin_labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Gene-level predicted delta (imputed - nucleus)")
ax.set_title("Distribution of Gene-Level Predicted Deltas by Broad Class")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
fig.tight_layout()
save_fig(fig, "human_m1_distributional_delta_summary")


# ──────────────────────────────────────────────────────────────────────
# F. Optional mouse-human projected visualization
# ──────────────────────────────────────────────────────────────────────
print("\n=== Part F: Mouse-human projected visualization ===")

# Load mouse per-CT pseudobulk deltas from delta tables
CT_NAME_MAP = {
    "L2_3 IT": "L2/3_IT", "L5 IT": "L5_IT", "L5 ET": "L5_ET",
    "L5_6 NP": "L5/6_NP", "L6 CT": "L6_CT", "L6 IT": "L6_IT",
    "L6 IT Car3": "L6_IT_Car3", "L6b": "L6b",
    "Pvalb": "Pvalb", "Sst": "Sst", "Vip": "Vip",
    "Lamp5": "Lamp5", "Sncg": "Sncg", "Sst Chodl": "Sst_Chodl",
    "Astro": "Astro", "Oligo": "Oligo", "OPC": "OPC",
    "Endo": "Endo", "VLMC": "VLMC", "SMC": "SMC",
    "Macrophage": "Micro",
}

mouse_delta_files = glob.glob(os.path.join(MOUSE_DELTA_DIR, "*_delta.csv"))
part_f_ok = len(mouse_delta_files) > 0

if part_f_ok:
    # build mouse delta matrix (CT x mapped_gene)
    mouse_cts = {}
    for fpath in mouse_delta_files:
        basename = os.path.basename(fpath).replace("_delta.csv", "")
        ct = CT_NAME_MAP.get(basename)
        if ct is None:
            continue
        gene_deltas = {}
        with open(fpath) as f:
            for row in csv.DictReader(f):
                gene_deltas[row["gene"]] = float(row["mean_delta"])
        mouse_cts[ct] = gene_deltas

    # human pseudobulk delta (mouse-like classes) from precomputed matrix
    human_delta_path = os.path.join(RES_DIR, "human_m1_predicted_delta_mouse_like_class_matrix.csv")
    human_delta_mat = pd.read_csv(human_delta_path, index_col=0)

    # find common CTs and genes
    common_cts = sorted(set(mouse_cts.keys()) & set(human_delta_mat.columns))
    mapped_gene_names = mouse_genes[mapped_idx]

    if len(common_cts) >= 3:
        # build aligned matrices: (n_cts, n_common_genes)
        common_genes_list = []
        for g in mapped_gene_names:
            if g in human_delta_mat.index and all(g in mouse_cts[ct] for ct in common_cts):
                common_genes_list.append(g)

        n_common = len(common_genes_list)
        print(f"  Common CTs: {len(common_cts)}, common genes: {n_common}")

        if n_common >= 100:
            mouse_mat = np.array([[mouse_cts[ct].get(g, 0) for g in common_genes_list] for ct in common_cts])
            human_mat = np.array([[human_delta_mat.loc[g, ct] for g in common_genes_list] for ct in common_cts])

            # PCA on stacked
            stacked = np.vstack([mouse_mat, human_mat])
            scaler_f = StandardScaler()
            stacked_s = scaler_f.fit_transform(stacked)
            pca_f = PCA(n_components=min(5, len(common_cts)), random_state=SEED)
            pca_f_coords = pca_f.fit_transform(stacked_s)
            n_ct = len(common_cts)
            mouse_coords = pca_f_coords[:n_ct]
            human_coords = pca_f_coords[n_ct:]

            fig, ax = plt.subplots(figsize=(10, 8))
            for i, ct in enumerate(common_cts):
                ax.scatter(mouse_coords[i, 0], mouse_coords[i, 1], c="blue", s=60, zorder=5, marker="o")
                ax.scatter(human_coords[i, 0], human_coords[i, 1], c="red", s=60, zorder=5, marker="^")
                ax.annotate("", xy=(human_coords[i, 0], human_coords[i, 1]),
                             xytext=(mouse_coords[i, 0], mouse_coords[i, 1]),
                             arrowprops=dict(arrowstyle="->", color="gray", lw=1))
                ax.text(mouse_coords[i, 0], mouse_coords[i, 1], ct, fontsize=7,
                        ha="right", va="bottom")

            ax.scatter([], [], c="blue", marker="o", label="Mouse predicted delta")
            ax.scatter([], [], c="red", marker="^", label="Human predicted delta")
            ax.set_xlabel(f"PC1 ({pca_f.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca_f.explained_variance_ratio_[1]:.1%})")
            ax.set_title(f"Mouse vs Human Predicted Delta (PCA on {n_common} shared genes, {len(common_cts)} CTs)")
            ax.legend(fontsize=9)
            fig.tight_layout()
            save_fig(fig, "human_m1_mouse_human_delta_projection")
            print("  Part F figure saved.")
        else:
            print(f"  Part F skipped: only {n_common} common genes (need >=100)")
            part_f_ok = False
    else:
        print(f"  Part F skipped: only {len(common_cts)} common CTs (need >=3)")
        part_f_ok = False
else:
    print("  Part F skipped: no mouse delta files found")


# ──────────────────────────────────────────────────────────────────────
# Generate report
# ──────────────────────────────────────────────────────────────────────
print("\n=== Generating report ===")

# gather summary stats
broad_identity = identity_df[identity_df["class_level"] == "broad_class"]
ml_identity = identity_df[identity_df["class_level"] == "mouse_like_class"]
broad_centroid = centroid_df[centroid_df["class_level"] == "broad_class"].sort_values("centroid_shift_distance", ascending=False)
ml_centroid = centroid_df[centroid_df["class_level"] == "mouse_like_class"].sort_values("centroid_shift_distance", ascending=False)
low_sample_classes = centroid_df[centroid_df["n_nuclei"] < 200][["class_level", "class", "n_nuclei"]]

# program score broad summary
ps_broad = program_df[program_df["class_level"] == "broad_class"]

report = f"""# Human M1 Global Imputation QC Report

**Date:** 2026-08-17
**Script:** `validation/human_m1_transfer/scripts/10_global_imputation_qc.py`

## 1. Input Files and Sample Sizes

| Input | Path | Shape/Size |
|-------|------|-----------|
| Nucleus matrix | `processed/human_m1_nuclei_mouse_model_order.npz` | {nuc_sparse.shape[0]} x {nuc_sparse.shape[1]} |
| Imputed matrix | `inference/human_m1_imputed_cell_log1p.npy` | {imputed.shape[0]} x {imputed.shape[1]} |
| Mapped genes | `processed/human_m1_available_gene_mask.npy` | {n_mapped} / {nuc_sparse.shape[1]} |
| Metadata | `inference/human_m1_inference_obs.csv` | {len(obs)} rows |

## 2. Subsampling

- Method: balanced by `mapped_mouse_like_class`, max {MAX_PER_CLASS} per class
- Seed: {SEED}
- Sampled: **{n_sampled}** nuclei from {len(unique_classes)} mouse-like classes
- Both raw nucleus and corresponding imputed profile included for each sampled nucleus

## 3. Gene Selection

- Mapped genes: {n_mapped}
- HVG method: top {n_hvg} by variance in raw nucleus matrix (sampled cells)
- Standardized before PCA

## 4. PCA/UMAP

- PCA: {N_PCA} components, explained variance: {pca.explained_variance_ratio_.sum():.3f}
- UMAP: {"computed (n_neighbors=30, min_dist=0.3)" if umap_available else "NOT AVAILABLE — only PCA plots produced"}

## 5. Centroid Shift Summary

### Top 5 broad classes by centroid shift distance:
"""

for _, row in broad_centroid.head(5).iterrows():
    flag = " ⚠️" if row["n_nuclei"] < 200 else ""
    report += f"- **{row['class']}**: shift={row['centroid_shift_distance']:.3f}, "
    report += f"mean_delta={row['mean_predicted_delta_all_mapped']:.4f}, n={row['n_nuclei']}{flag}\n"

report += "\n### Top 5 mouse-like classes by centroid shift distance:\n"
for _, row in ml_centroid.head(5).iterrows():
    flag = " ⚠️" if row["n_nuclei"] < 200 else ""
    report += f"- **{row['class']}**: shift={row['centroid_shift_distance']:.3f}, "
    report += f"mean_delta={row['mean_predicted_delta_all_mapped']:.4f}, n={row['n_nuclei']}{flag}\n"

report += "\n### Bottom 3 mouse-like classes by centroid shift distance:\n"
for _, row in ml_centroid.tail(3).iterrows():
    flag = " ⚠️" if row["n_nuclei"] < 200 else ""
    report += f"- **{row['class']}**: shift={row['centroid_shift_distance']:.3f}, "
    report += f"mean_delta={row['mean_predicted_delta_all_mapped']:.4f}, n={row['n_nuclei']}{flag}\n"

report += f"""
## 6. Cell Identity Preservation

### Broad class:
| Class | n | Raw self-assign | Imputed → original | Drop |
|-------|---|----------------|-------------------|------|
"""
for _, row in broad_identity.sort_values("imputed_assignment_to_original_class_rate").iterrows():
    report += f"| {row['class']} | {row['n_sampled']} | {row['raw_self_assignment_rate']:.3f} | "
    report += f"{row['imputed_assignment_to_original_class_rate']:.3f} | {row['drop_from_raw']:.3f} |\n"

report += f"""
### Mouse-like class (summary):
- Median imputed → original rate: **{ml_identity['imputed_assignment_to_original_class_rate'].median():.3f}**
- Min: {ml_identity['imputed_assignment_to_original_class_rate'].min():.3f} ({ml_identity.loc[ml_identity['imputed_assignment_to_original_class_rate'].idxmin(), 'class']})
- Max: {ml_identity['imputed_assignment_to_original_class_rate'].max():.3f} ({ml_identity.loc[ml_identity['imputed_assignment_to_original_class_rate'].idxmax(), 'class']})

## 7. Program Score Shifts (Broad Class)

| Class | SynGO | TF | Ribosomal | Nuclear lncRNA |
|-------|-------|-----|-----------|----------------|
"""

for cls in broad_cls_order:
    vals = {}
    for gs in ["SynGO", "TF", "Ribosomal", "Nuclear_lncRNA"]:
        match = ps_broad[(ps_broad["class"] == cls) & (ps_broad["gene_set"] == gs)]
        vals[gs] = f"{match['score_delta_mean'].values[0]:.4f}" if len(match) > 0 else "n/a"
    report += f"| {cls} | {vals['SynGO']} | {vals['TF']} | {vals['Ribosomal']} | {vals['Nuclear_lncRNA']} |\n"

# gene counts
gs_counts = {}
for gs in ["SynGO", "TF", "Ribosomal", "Nuclear_lncRNA"]:
    match = ps_broad[ps_broad["gene_set"] == gs]
    gs_counts[gs] = match["n_genes_used"].values[0] if len(match) > 0 else 0

report += f"""
Gene counts: SynGO={gs_counts['SynGO']}, TF={gs_counts['TF']}, Ribosomal={gs_counts['Ribosomal']}, Nuclear_lncRNA={gs_counts['Nuclear_lncRNA']}

## 8. Distributional Delta Summary (Broad Class)

| Class | n_nuclei | Mean delta | Median delta | % positive | % large (>0.5) | IQR |
|-------|---------|-----------|-------------|-----------|---------------|-----|
"""
for _, row in broad_dist.sort_values("mean_delta", ascending=False).iterrows():
    report += (f"| {row['class']} | {row['n_nuclei']} | {row['mean_delta']:.4f} | "
               f"{row['median_delta']:.4f} | {row['fraction_positive_delta']:.1%} | "
               f"{row['fraction_large_positive_delta']:.1%} | {row['delta_iqr']:.4f} |\n")

report += f"""
## 9. Low-Sample Warnings

Classes with fewer than 200 sampled nuclei:
"""
if len(low_sample_classes) > 0:
    for _, row in low_sample_classes.iterrows():
        report += f"- {row['class_level']}: **{row['class']}** (n={row['n_nuclei']})\n"
else:
    report += "None.\n"

report += f"""
## 10. Mouse-Human Delta Projection (Part F)

{"Figure saved: `human_m1_mouse_human_delta_projection.png/pdf`" if part_f_ok else "Skipped — insufficient common cell types or genes between mouse validation and human transfer."}

## 11. Interpretation and Caveats

1. **No human whole-cell ground truth.** Allen Human M1 data are nuclei only. We cannot validate whether imputed profiles match actual human whole-cell expression.
2. **Zero-shot cross-species transfer.** The model was trained exclusively on mouse MOp data.
3. **Ortholog-restricted genes only.** Only {n_mapped}/{nuc_sparse.shape[1]} model genes have detected one-to-one human orthologs. Unmapped genes are zero-filled and excluded from all analyses here.
4. **Cell-type mapping is approximate** at the fine-subclass level (human M1 subclasses named to match mouse taxonomy).
5. **SynGO annotates protein function/location, not RNA localization.** The SynGO score shift reflects that genes encoding synaptic proteins are more cytoplasmically enriched in neurons, but the causal link is protein function, not direct RNA transport.
6. **These are global QC/plausibility checks, not direct human validation.**

## 12. Summary

Global QC shows whether the frozen mouse-trained model shifts human M1 nuclei toward cell-like profiles while preserving cell identity and biological program structure. Because no human whole-cell profiles are available, these analyses support cross-species plausibility rather than direct human accuracy.

## Output Files

```
figures/
  human_m1_global_pca_umap.{{png,pdf}}
  human_m1_centroid_arrows.{{png,pdf}}
  human_m1_identity_preservation.{{png,pdf}}
  human_m1_program_score_shifts.{{png,pdf}}
  human_m1_distributional_delta_summary.{{png,pdf}}
{"  human_m1_mouse_human_delta_projection.{png,pdf}" if part_f_ok else "  (mouse-human projection skipped)"}
results/
  human_m1_global_embedding.csv
  human_m1_centroid_shift_summary.csv
  human_m1_identity_preservation.csv
  human_m1_program_score_shifts.csv
  human_m1_distributional_delta_summary.csv
reports/
  HUMAN_M1_GLOBAL_QC_REPORT.md
```
"""

with open(os.path.join(REP_DIR, "HUMAN_M1_GLOBAL_QC_REPORT.md"), "w") as f:
    f.write(report)
print("  Saved HUMAN_M1_GLOBAL_QC_REPORT.md")

print("\n=== Done ===")
