"""Per-cell expression densities of selected genes in one cell type: real cells, raw nuclei, and the imputed decodes
with and without the library-size covariate (2026-11-13, asked after the Prashant 2026-11-06 meeting).

Series: real cells; raw nuclei; old model (no covariate, cvae_seed1); covariate model decoded at each nucleus's quantile-mapped library (qmap); the same + Poisson sampling
of the decoded counts. All on log1p counts, test partition.
Usage: python analysis/benchmark/gene_density_ct.py --ct "L5 IT"
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); from data_split import load_partition, load_gene_names

BASE = "run/benchmark/test"
GENES = [("Pcp4", "cytoplasmic, absent in nuclei"), ("Snap25", "synaptic, cytoplasm-enriched"), ("Rplp1", "ribosomal, absent in nuclei"),
         ("Rorb", "L4/L5 IT marker"), ("Fezf2", "L5 ET marker, low and patchy in other types"), ("Calm2", "housekeeping, >70% detected"),
         ("Malat1", "nuclear lncRNA"), ("Xist", "sex gene, top real-cell variance"), ("Gm49164", "lncRNA, heavy tail in real cells"),
         ("Gfra1", "sparse (<10% detected)")]
SERIES = [("real cells", "k", "-"), ("raw nuclei", "gray", "--"), ("old model (no covariate)", "#9ecae1", "-"),
          ("covariate, per-cell library", "#1f77b4", "-"),
          ("covariate, per-cell library + Poisson", "#ff7f0e", "-")]


def type_specific_genes(Xc, ct_all, R, N, genes, exclude, n_pos=3, n_neg=2, n_marker=2):
    """Per-type additions (test partition, real cells R and nuclei N of the type; Xc = all real test cells):
    largest positive / negative cell-minus-nucleus shift (detection >= 30 % in the compartment that is higher),
    top markers of the type vs all other types (real cells), one heavy-tailed gene (highest count var/mean among
    genes detected in 30-70 % of real cells), one sparse gene (detection 3-10 %, highest variance)."""
    mc, mn = R.mean(0), N.mean(0); dc, dn = (R > 0.05).mean(0), (N > 0.05).mean(0)
    shift = mc - mn; order = np.argsort(-shift); out = []
    pos = [j for j in order if dc[j] >= 0.3 and genes[j] not in exclude][:n_pos]
    neg = [j for j in order[::-1] if dn[j] >= 0.3 and genes[j] not in exclude][:n_neg]
    out += [(genes[j], f"type-specific: top +shift (cell - nucleus {shift[j]:+.2f})") for j in pos]
    out += [(genes[j], f"type-specific: top -shift (cell - nucleus {shift[j]:+.2f})") for j in neg]
    return out


def _markers_and_tails(Xc, ct_all, ct, R, genes, exclude, n_marker=2):
    mask = ct_all == ct
    diff = Xc[mask].mean(0) - Xc[~mask].mean(0)
    mk = [j for j in np.argsort(-diff) if genes[j] not in exclude][:n_marker]
    counts = np.expm1(R); det = (R > 0.05).mean(0); vm = counts.var(0) / np.maximum(counts.mean(0), 1e-6)
    tail_pool = np.where((det >= 0.3) & (det <= 0.7))[0]; tail = [j for j in tail_pool[np.argsort(-vm[tail_pool])] if genes[j] not in exclude][:1]
    sp_pool = np.where((det >= 0.03) & (det < 0.1))[0]; sparse = [j for j in sp_pool[np.argsort(-R.var(0)[sp_pool])] if genes[j] not in exclude][:1]
    out = [(genes[j], f"type-specific: marker of {ct} (vs others {diff[j]:+.2f})") for j in mk]
    out += [(genes[j], f"type-specific: heavy tail (count var/mean {vm[j]:.1f}, det {det[j]*100:.0f}%)") for j in tail]
    out += [(genes[j], f"type-specific: sparse (det {det[j]*100:.0f}%)") for j in sparse]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ct", default="L5 IT"); ap.add_argument("--out", default="notes/experiments/libsize_covariate/armA/figures")
    ap.add_argument("--extra", default="", help="extra genes as gene:label;gene:label (label text; files prefixed ex_)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    genes = list(load_gene_names()); gi = {g: i for i, g in enumerate(genes)}
    Xc, oc = load_partition("cell", "test", "log1p"); Xn, on = load_partition("nuc", "test", "log1p")
    data = {"real cells": Xc[oc.cell_type.values == a.ct], "raw nuclei": Xn[on.cell_type.values == a.ct]}
    fixed = {g for g, _ in GENES}
    extra = type_specific_genes(Xc, oc.cell_type.values, data["real cells"], data["raw nuclei"], genes, fixed)
    extra += _markers_and_tails(Xc, oc.cell_type.values, a.ct, data["real cells"], genes, fixed | {g for g, _ in extra})
    GENES_CT = list(GENES) + extra
    print("type-specific genes:", [g for g, _ in extra])
    if a.extra:
        GENES_CT += [(x.split(":")[0].strip(), "extra: " + x.split(":", 1)[1].strip()) for x in a.extra.split(";") if x.strip()]
    for name, d in [("old model (no covariate)", "cvae_seed1"), ("covariate, fixed library", "cvae_libsize_armA"), ("covariate, per-cell library", "cvae_libsize_armA_qmap")]:
        im = np.load(f"{BASE}/{d}/imputed_eval.npy", mmap_mode="r"); ob = pd.read_csv(f"{BASE}/{d}/nuc_eval_obs.csv")
        data[name] = np.asarray(im[ob.cell_type.values == a.ct])
    rng = np.random.default_rng(0)
    data["covariate, per-cell library + Poisson"] = np.log1p(rng.poisson(np.expm1(np.clip(data["covariate, per-cell library"], 0, None))))
    nrow = int(np.ceil(len(GENES_CT) / 5))
    fig, axes = plt.subplots(nrow, 5, figsize=(26, 4.5 * nrow)); axes = axes.ravel()  # overview: five series (fixed-library decode dropped 2026-11-13)
    for ax in axes[len(GENES_CT):]: ax.axis("off")
    for ax, (g, desc) in zip(axes, GENES_CT):
        j = gi[g]; hi = max(v[:, j].max() for v in data.values()); bins = np.linspace(0, hi + 0.1, 45); txt = []
        for name, col, ls in SERIES:
            v = data[name][:, j]
            ax.hist(v, bins=bins, density=True, histtype="step", color=col, ls=ls, lw=1.8 if name == "real cells" else 1.4, label=name)
            txt.append(f"{name[:26]:26s} var {v.var():.2f}  det {(v > 0.05).mean()*100:3.0f}%")
        ax.set_title(f"{g}\n{desc}", fontsize=10); ax.set_xlabel("log1p counts"); ax.set_ylim(0, min(ax.get_ylim()[1], 3.5))
        ax.text(0.98, 0.97, "\n".join(txt), transform=ax.transAxes, ha="right", va="top", fontsize=6.5, family="monospace",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    for k in range(0, len(GENES_CT), 5): axes[k].set_ylabel("density")
    axes[0].legend(fontsize=7, loc="center right")
    nc, nn = len(data["real cells"]), len(data["raw nuclei"])
    fig.suptitle(f"{a.ct}, test partition ({nc:,} real cells, {nn:,} nuclei): per-cell expression density, with and without the library-size covariate", fontsize=12)
    fig.tight_layout(); tag = a.ct.replace(" ", "").replace("/", "")
    fig.savefig(f"{a.out}/fig_gene_densities_{tag}_overview.png", dpi=110); print("wrote", f"{a.out}/fig_gene_densities_{tag}_overview.png")
    plt.close(fig)
    # one figure per gene: overlay + one panel per series, shared x and y scales
    gdir = f"{a.out}/gene_densities_{tag}"; os.makedirs(gdir, exist_ok=True)
    for g, desc in GENES_CT:
        j = gi[g]; hi = max(v[:, j].max() for v in data.values()); bins = np.linspace(0, hi + 0.1, 45)
        hists = {name: np.histogram(data[name][:, j], bins=bins, density=True)[0] for name, _, _ in SERIES}
        ymax = 1.1 * max(h.max() for h in hists.values())
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True); axes = axes.ravel()
        ax = axes[0]
        for name, col, ls in SERIES:
            ax.hist(data[name][:, j], bins=bins, density=True, histtype="step", color=col, ls=ls, lw=1.8 if name == "real cells" else 1.4, label=name)
        ax.set_title("all series", fontsize=11); ax.legend(fontsize=8)
        for ax, (name, col, ls) in zip(axes[1:], SERIES):
            v = data[name][:, j]
            ax.hist(v, bins=bins, density=True, histtype="stepfilled", color=col, alpha=0.45, edgecolor=col, lw=1.4)
            ax.hist(data["real cells"][:, j], bins=bins, density=True, histtype="step", color="k", ls=":", lw=1, label="real cells (reference)")
            ax.set_title(name, fontsize=11)
            ax.text(0.98, 0.97, f"mean {v.mean():.2f}  sd {v.std():.2f}\nvar {v.var():.2f}  detected {(v > 0.05).mean()*100:.0f}%\nn = {len(v):,}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9, family="monospace", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        axes[1].legend(fontsize=8, loc="center right")
        for ax in axes:
            ax.set_ylim(0, ymax); ax.set_xlim(bins[0], bins[-1])
        for ax in axes[3:]: ax.set_xlabel("log1p counts")
        axes[0].set_ylabel("density"); axes[3].set_ylabel("density")
        fig.suptitle(f"{g}: {desc}   ({a.ct}, test partition; shared axes)", fontsize=13)
        fname = f"ct_{g}" if desc.startswith("type-specific") else (f"ex_{g}" if desc.startswith("extra") else g)
        fig.tight_layout(); fig.savefig(f"{gdir}/{fname}.png", dpi=110); plt.close(fig)
    print("wrote per-gene figures to", gdir)


if __name__ == "__main__":
    main()
