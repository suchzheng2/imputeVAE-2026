#!/usr/bin/env python3
"""L5 cell-type mapping audit (Prashant, Sep 2026): do not map external L5 TRAP populations to L5 IT / L5 ET by name.

For each external layer-5 (and the L4) population:
  (A) each L5 type individually: H2 profile Spearman (nucleus / imputed / real cell) against L5 IT, L5 ET, L5/6 NP and the
      other excitatory types, on the population's own expressed genes; uncentered and centered (deviation from the
      excitatory mean on both sides, external side centred on the mean of all external excitatory populations).
  (B) marker-gene scoring, using REAL-CELL pseudobulk only (never imputed data, so the choice of match is not circular):
      data-derived markers (top genes specific to each L5 type among the 7 excitatory types) and a canonical Allen panel.
      Score = mean z (within the external profile, across expressed genes) of the marker set; assignment = argmax.
  (C) Scnn1a_L4 is reported but flagged EXCLUDED: motor cortex has no L4.
Outputs: outputs/extension/tables/l5_audit_*.tsv, outputs/extension/figures/L5_audit_*.png
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
s08 = import_module("08_profile_improvement"); s15 = import_module("15_extension_h2_h3")
TASK = s08.TASK; EXT = f"{TASK}/outputs/extension"; TAB = f"{EXT}/tables"; FIG = f"{EXT}/figures"; os.makedirs(FIG, exist_ok=True)
EXC = ['L2/3 IT', 'L5 IT', 'L5 ET', 'L5/6 NP', 'L6 CT', 'L6 IT', 'L6b']; L5 = ['L5 IT', 'L5 ET', 'L5/6 NP']
L5_POPS = ["Gprin3_L5b", "Colgalt2_L5b", "S100a10_L5a", "Glt25d2_L5b", "Rbp4_L5", "Scnn1a_L4"]
EXCLUDED = {"Scnn1a_L4": "layer 4 driver; motor cortex has no granular layer -> no counterpart, excluded from H2/H3"}
# Canonical markers with a verified literature source (see outputs/extension/metadata/l5_mapping_evidence.md), present in the 10,134-gene space:
# L5 ET: Fezf2, Bcl11b (Harris & Shepherd 2015; Yao 2021), Bcl6 (Tasic 2018), Npnt / Slco2a1 (Yao 2021 MOp, non-MY / MY-projecting ET); Fam84b absent.
# L5 IT: Rorb (Yao 2021 MOp), Slc30a3 (pan-IT, Tasic 2018), Cux2 (Tasic 2018 layer marker), Rspo1 (L4/5 IT MOp, Yao 2021).
# L5/6 NP: Tshz2 (Tasic 2018; Yao 2021); Slc17a8 / Trhr / Sla2 / Rapgef3 absent from the gene space.
CANON = {"L5 ET": ["Fezf2", "Bcl11b", "Bcl6", "Npnt", "Slco2a1"], "L5 IT": ["Rorb", "Slc30a3", "Cux2", "Rspo1"], "L5/6 NP": ["Tshz2"]}
SEED = 42; N_BOOT = 300


def sp(a, b): return float(stats.spearmanr(a, b)[0])


def main():
    rng = np.random.default_rng(SEED); m = s15.load_model()
    pops = [p for p in s15.populations(include_excluded=True) if p["cell_class"] == "excitatory"]   # keep Scnn1a_L4 so the exclusion is shown
    # one profile per population: META where available, else the single batch
    prof = {}
    for p in pops:
        k = p["population"]
        if p.get("is_meta") or k not in prof: prof[k] = p
    cell = m["cell"]
    # ---------------- (B1) data-derived markers from real cells: specificity = target - max(other excitatory types), among expressed genes
    markers = {}
    for ct in L5:
        others = [c for c in EXC if c != ct]
        spec = (cell[ct] - cell[others].max(axis=1)); spec = spec[cell[ct] > 1.0]
        markers[ct] = spec.sort_values(ascending=False).head(40)
    pd.concat([pd.DataFrame({"target": ct, "gene": s.index, "specificity_log1p": s.values, "mean_target": cell.loc[s.index, ct].values}) for ct, s in markers.items()]).to_csv(f"{TAB}/l5_audit_markers_realcell.tsv", sep="\t", index=False)
    canon_tab = pd.DataFrame({g: cell.loc[g, EXC] for ct in CANON for g in CANON[ct] if g in cell.index}).T
    canon_tab.index.name = "gene"; canon_tab.to_csv(f"{TAB}/l5_audit_canonical_markers_realcell_pseudobulk.tsv", sep="\t"); print("canonical markers, real-cell log1p pseudobulk:\n", canon_tab.round(2).to_string())
    # ---------------- (A) each excitatory type individually
    rowsA, rowsC, rowsB = [], [], []
    # external centring reference: mean of all external excitatory profiles on their common expressed genes
    common = None
    for k, p in prof.items():
        mu = p["reps"].mean(axis=1); g = mu.index[mu > p["thr"]]; common = g if common is None else common.intersection(g)
    common = common.intersection(m["nucleus"].index)
    E = pd.concat([p["reps"].loc[common].mean(axis=1).rename(k) for k, p in prof.items()], axis=1); Ec = E.sub(E.mean(axis=1), axis=0)
    for k in L5_POPS:
        p = prof[k]; mu = p["reps"].mean(axis=1); genes = mu.index[mu > p["thr"]].intersection(m["nucleus"].index)
        for ct in EXC:
            st = s08.profile_stats(mu, m, ct, genes, rng, N_BOOT)
            rowsA.append(dict(population=k, dataset=p["dataset"], batch=p["batch"], model_ct=ct, is_l5=ct in L5, name_based_match=p["model_col"], excluded=k in EXCLUDED, n_genes=len(genes),
                              spearman_nucleus=st["spearman_nucleus"], spearman_imputed=st["spearman_imputed"], spearman_realcell=st["spearman_realcell"],
                              imp_minus_nuc=st["delta_similarity_spearman"], imp_minus_nuc_ci_lo=st["gene_boot_ci_lo"], imp_minus_nuc_ci_hi=st["gene_boot_ci_hi"], gap_closed_frac=st["gap_closed_frac"]))
            # centred: which type's deviation from the excitatory mean does this population's deviation follow?
            for kind in ("nucleus", "imputed", "cell"):
                M = m[kind].loc[common, EXC]; Mc = M.sub(M.mean(axis=1), axis=0)
                rowsC.append(dict(population=k, model_ct=ct, kind=kind, spearman_centered=sp(Mc[ct], Ec[k]), spearman_model_centered_only=sp(Mc[ct], E[k]), n_genes=len(common)))
        # ---------------- (B2) marker scores within the external profile
        z = (mu.loc[genes] - mu.loc[genes].mean()) / mu.loc[genes].std()
        for src, sets in (("data_derived_realcell", markers), ("canonical", CANON)):
            for ct, gl in sets.items():
                gl = [g for g in (gl.index if isinstance(gl, pd.Series) else gl) if g in z.index]
                rowsB.append(dict(population=k, marker_source=src, marker_set=ct, n_markers_present=len(gl), mean_z=float(z.loc[gl].mean()), median_z=float(z.loc[gl].median()),
                                  frac_above_median=float((mu.loc[gl] > mu.loc[genes].median()).mean())))
        for ct, gl in CANON.items():
            for g in gl:
                if g in z.index: rowsB.append(dict(population=k, marker_source="canonical_gene", marker_set=ct, gene=g, n_markers_present=1, mean_z=float(z.loc[g]), median_z=float(z.loc[g]), frac_above_median=np.nan))
    A = pd.DataFrame(rowsA); C = pd.DataFrame(rowsC); B = pd.DataFrame(rowsB)
    A.to_csv(f"{TAB}/l5_audit_each_type_H2.tsv", sep="\t", index=False); C.to_csv(f"{TAB}/l5_audit_each_type_centered.tsv", sep="\t", index=False); B.to_csv(f"{TAB}/l5_audit_marker_scores.tsv", sep="\t", index=False)
    # ranks / assignment summary
    summ = []
    for k in L5_POPS:
        a = A[(A.population == k) & A.is_l5].set_index("model_ct"); c = C[(C.population == k) & C.model_ct.isin(L5)]
        b = B[(B.population == k) & (B.marker_source != "canonical_gene")]
        row = dict(population=k, dataset=prof[k]["dataset"], name_based_match=prof[k]["model_col"], excluded=k in EXCLUDED, exclusion_reason=EXCLUDED.get(k, ""))
        for kind in ("realcell", "imputed", "nucleus"):
            row[f"best_l5_uncentered_{kind}"] = a[f"spearman_{kind}"].idxmax(); row[f"spread_l5_uncentered_{kind}"] = float(a[f"spearman_{kind}"].max() - a[f"spearman_{kind}"].min())
        for kind in ("cell", "imputed", "nucleus"):
            cc = c[c.kind == kind].set_index("model_ct").spearman_centered; row[f"best_l5_centered_{kind}"] = cc.idxmax(); row[f"centered_L5IT_{kind}"] = cc["L5 IT"]; row[f"centered_L5ET_{kind}"] = cc["L5 ET"]; row[f"centered_NP_{kind}"] = cc["L5/6 NP"]
        for src in ("data_derived_realcell", "canonical"):
            bb = b[b.marker_source == src].set_index("marker_set").mean_z; row[f"marker_best_{src}"] = bb.idxmax(); row[f"marker_z_ET_minus_IT_{src}"] = bb["L5 ET"] - bb["L5 IT"]
        summ.append(row)
    S = pd.DataFrame(summ); S.to_csv(f"{TAB}/l5_audit_summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 300)
    print("\n== each L5 type individually (uncentered H2 Spearman) ==")
    print(A[A.is_l5].pivot_table(index="population", columns="model_ct", values=["spearman_nucleus", "spearman_imputed", "spearman_realcell"]).round(3).to_string())
    print("\n== centred (both sides), L5 types ==")
    print(C[C.model_ct.isin(L5)].pivot_table(index=["population", "kind"], columns="model_ct", values="spearman_centered").round(3).to_string())
    print("\n== marker scores (mean z within external profile) ==")
    print(B[B.marker_source != "canonical_gene"].pivot_table(index=["population", "marker_source"], columns="marker_set", values="mean_z").round(2).to_string())
    print("\n== canonical genes, z within external profile ==")
    print(B[B.marker_source == "canonical_gene"].pivot_table(index="population", columns="gene", values="mean_z").round(2).to_string())
    print("\n== summary ==\n", S[["population", "name_based_match", "excluded", "best_l5_centered_cell", "best_l5_centered_nucleus", "best_l5_centered_imputed", "marker_best_data_derived_realcell", "marker_best_canonical", "marker_z_ET_minus_IT_data_derived_realcell", "marker_z_ET_minus_IT_canonical"]].to_string())
    # ---------------- figure: centred Spearman heatmaps (7 excitatory types) for real cell / nucleus / imputed + marker ET-IT score
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15, 0.7]))
    for ax, kind in zip(axes[:3], ("cell", "nucleus", "imputed")):
        H = C[C.kind == kind].pivot_table(index="population", columns="model_ct", values="spearman_centered").loc[L5_POPS, EXC]
        im = ax.imshow(H.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_xticks(range(len(EXC))); ax.set_xticklabels(EXC, rotation=45, ha="right"); ax.set_yticks(range(len(L5_POPS))); ax.set_yticklabels([f"{p}{' (excluded)' if p in EXCLUDED else ''}" for p in L5_POPS])
        for i in range(H.shape[0]):
            j = int(np.nanargmax(H.values[i])); ax.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", fontsize=7, fontweight="bold")
        ax.set_title(f"external vs model {kind}\n(centred Spearman, n={int(C.n_genes.iloc[0])} genes)", fontsize=9); ax.grid(False)
    fig.colorbar(im, ax=axes[2], fraction=0.05, pad=0.03, label="centred Spearman")
    ax = axes[3]; bb = B[B.marker_source != "canonical_gene"].pivot_table(index=["population", "marker_source"], columns="marker_set", values="mean_z")
    y = np.arange(len(L5_POPS))
    for off, src, col in ((-0.18, "data_derived_realcell", "#0072B2"), (0.18, "canonical", "#E69F00")):
        v = [bb.loc[(p, src), "L5 ET"] - bb.loc[(p, src), "L5 IT"] for p in L5_POPS]; ax.barh(y + off, v, 0.36, color=col, label=f"{src} markers")
    ax.axvline(0, color="#7F7F7F", lw=0.8); ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_xlabel("marker z: L5 ET minus L5 IT\n(>0 = ET-like)"); ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("marker scoring (real-cell markers)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/L5_audit_centered_and_markers.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{FIG}/L5_audit_centered_and_markers.pdf", bbox_inches="tight")
    print("saved figure")


if __name__ == "__main__":
    main()
