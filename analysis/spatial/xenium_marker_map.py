"""Spatial marker map of the Xenium Prime 5K coronal section (2026-11-13): settles the cortical area / bregma level.

Panel A: coarse region of each cell from the 10x graph clusters (striatum / thalamus / hippocampus vs the rest).
Panels B-F: per-cell log1p(CP100) of layer markers along the cortical sheet. Rorb marks the granular L4 (strong in
somatosensory cortex, weak/absent in motor cortex); Cux2 L2/3, Fezf2 L5 ET, Syt6 L6 CT, Ppp1r1b striatum.
Static PNG for the notes (sequential single-hue ramp; three categorical hues + grey).
"""
import os, numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from xenium_scoping import read_10x_h5, XEN, OUT

SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = {"Striatum": "#2a78d6", "Thalamus": "#eb6834", "Hippocampus": "#1baf7a", "other": "#d8d7d0"}
REGION_CLUSTERS = {"9": "Striatum", "7": "Thalamus", "17": "Hippocampus"}  # from cluster_markers.csv
GENES = ["Rorb", "Cux2", "Fezf2", "Syt6", "Ppp1r1b"]


def main():
    X, names, ftype, bc = read_10x_h5(f"{XEN}/cell_feature_matrix.h5")
    X, names = X[:, ftype == "Gene Expression"], names[ftype == "Gene Expression"]
    gi = {g: i for i, g in enumerate(names)}
    obs = pd.read_csv(f"{OUT}/cell_obs_with_cluster.csv.gz", index_col=0).reindex(bc)
    ok = obs.transcript_counts.notna().values
    X, obs = X[ok], obs[ok]
    tot = np.asarray(X.sum(1)).ravel()
    obs["region"] = obs.cluster.astype(float).astype(int).astype(str).map(REGION_CLUSTERS).fillna("other")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), facecolor="#fcfcfb")
    ax = axes[0, 0]
    for r, col in CAT.items():
        s = obs[obs.region == r]
        ax.scatter(s.x_centroid, s.y_centroid, s=0.6, c=col, lw=0, label=f"{r} (n={len(s):,})", rasterized=True)
    ax.set_title("A  Region from 10x clusters", loc="left", fontsize=11, color="#0b0b0b")
    ax.legend(markerscale=12, frameon=False, fontsize=8, loc="lower left")
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    for ax, g, lab in zip(axes.ravel()[1:], GENES, ["B", "C", "D", "E", "F"]):
        v = np.log1p(np.asarray(X[:, gi[g]].todense()).ravel() * 100.0 / tot)
        o = np.argsort(v)
        ax.scatter(obs.x_centroid.values[o], obs.y_centroid.values[o], s=0.6, c=v[o], cmap=cmap, lw=0, vmin=0,
                   vmax=np.percentile(v, 99), rasterized=True)
        ax.set_title(f"{lab}  {g}  log1p(CP100), 99th pct cap", loc="left", fontsize=11, color="#0b0b0b")
    for ax in axes.ravel():
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_facecolor("#fcfcfb")
        ax.tick_params(colors="#898781", labelsize=7); [s.set_color("#c3c2b7") for s in ax.spines.values()]
        ax.set_xlabel("x (um)", color="#898781", fontsize=8); ax.set_ylabel("y (um)", color="#898781", fontsize=8)
    fig.suptitle("Xenium Prime 5K mouse brain coronal section: regions and cortical layer markers", x=0.01, ha="left",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout()
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    fig.savefig(f"{OUT}/figures/fig_xenium_region_layer_markers.png", dpi=130)
    print("wrote", f"{OUT}/figures/fig_xenium_region_layer_markers.png")


if __name__ == "__main__":
    main()
