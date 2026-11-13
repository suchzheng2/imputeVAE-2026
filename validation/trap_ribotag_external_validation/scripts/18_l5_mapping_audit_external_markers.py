#!/usr/bin/env python3
"""L5 mapping audit with an EXTERNAL marker list (Yao 2021 MOp, Supplementary Table 6), 2026-09-25.

Why: script 17 scored markers derived from this repo's own real-cell pseudobulk, which is also the H2 "real whole cell"
ceiling, so the check was not independent of a reported number. Here nothing from this repo's data enters the marker
definition; the only project inputs are (i) the external TRAP profiles and (ii) the 10,134-gene model space used as the
gene universe. Model pseudobulks (nucleus / imputed / real cell) are NOT loaded.

Markers: outputs/extension/metadata/yao2021/yao2021_l5_subclass_marker_fractions.tsv (see README there). A gene is a
marker of subclass X if it is higher in X clusters than in the clusters of the other two L5 subclasses in >= 80% of
cluster pairs and higher than all other glutamatergic clusters in >= 50% of pairs.

Scoring, for each external L5-audit population (Gprin3, Colgalt2, S100a10, Glt25d2, Rbp4, Scnn1a):
  (1) cross-population z: on genes expressed in every external glutamatergic population, z-score each gene ACROSS the
      external populations; set score = mean z over the marker set. Asks "is Bcl6 higher in Gprin3 than in S100a10?",
      which is the question. Primary result.
  (2) within-profile z (as in script 17, for comparability): z across genes inside one profile; mostly abundance.
Assignment = argmax of (1) over {L5 IT, L5 ET, L5/6 NP}; compared with the literature-based mapping in
extension_populations.tsv (Rbp4 = POOL_L5_IT_ET is "mixed").
Outputs: outputs/extension/tables/l5_audit_extmarkers_*.tsv, outputs/extension/figures/L5_audit_external_markers.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
s15 = import_module("15_extension_h2_h3")
TASK = s15.TASK; EXT = f"{TASK}/outputs/extension"; TAB = f"{EXT}/tables"; FIG = f"{EXT}/figures"; META = f"{EXT}/metadata"
L5 = ["L5 IT", "L5 ET", "L5/6 NP"]
L5_POPS = ["Gprin3_L5b", "Colgalt2_L5b", "S100a10_L5a", "Glt25d2_L5b", "Rbp4_L5", "Scnn1a_L4"]
LIT = {"Gprin3_L5b": "L5 ET", "Colgalt2_L5b": "L5 ET", "S100a10_L5a": "L5 IT", "Glt25d2_L5b": "L5 ET", "Rbp4_L5": "mixed IT+ET", "Scnn1a_L4": "L4 (excluded)"}
EXCL = pd.read_csv(f"{META}/extension_populations.tsv", sep="\t").query("exclude == True").set_index("population").exclude_reason.to_dict()
FRAC_L5, FRAC_EXC = 0.8, 0.5


def main():
    gene_space = pd.Index(pd.read_csv(f"{TASK.rsplit('/validation', 1)[0]}/data/training_gene_list_v3.csv").iloc[:, 0].astype(str))
    F = pd.read_csv(f"{META}/yao2021/yao2021_l5_subclass_marker_fractions.tsv", sep="\t")
    sel = F[(F.frac_up_vs_other_L5 >= FRAC_L5) & (F.frac_up_vs_other_excitatory >= FRAC_EXC)]
    markers = {ct: sel[sel.subclass == ct].sort_values("frac_up_vs_other_excitatory", ascending=False) for ct in L5}
    mk = pd.concat([m.assign(in_model_gene_space=m.gene.isin(gene_space)) for m in markers.values()])
    mk.to_csv(f"{TAB}/l5_audit_extmarkers_yao2021_sets.tsv", sep="\t", index=False)
    print("Yao 2021 marker sets (all / in model gene space):", {ct: (len(m), int(m.gene.isin(gene_space).sum())) for ct, m in markers.items()})
    # external glutamatergic profiles, one per population (META where available)
    pops = [p for p in s15.populations(include_excluded=True) if p["cell_class"] == "excitatory"]
    prof = {}
    for p in pops:
        k = p["population"]
        if p.get("is_meta") or k not in prof: prof[k] = p
    mu = {k: p["reps"].mean(axis=1) for k, p in prof.items()}
    expressed = {k: mu[k].index[mu[k] > p["thr"]] for k, p in prof.items()}
    common = None
    for k in prof: common = expressed[k] if common is None else common.intersection(expressed[k])
    common = common.intersection(gene_space)
    E = pd.concat([mu[k].loc[common].rename(k) for k in prof], axis=1)
    Z = E.sub(E.mean(axis=1), axis=0).div(E.std(axis=1, ddof=1), axis=0)          # cross-population z, per gene
    print(f"external glutamatergic populations: {list(E.columns)}; common expressed genes in model space: {len(common)}")
    rows, gene_rows = [], []
    for k in L5_POPS:
        genes = expressed[k].intersection(gene_space)
        zin = (mu[k].loc[genes] - mu[k].loc[genes].mean()) / mu[k].loc[genes].std()
        for ct in L5:
            g_all = [g for g in markers[ct].gene if g in gene_space]
            g_x = [g for g in g_all if g in Z.index]; g_w = [g for g in g_all if g in zin.index]
            rows.append(dict(population=k, marker_set=ct, n_markers=len(g_all), n_present_cross=len(g_x), n_present_within=len(g_w),
                             cross_pop_mean_z=float(Z.loc[g_x, k].mean()), cross_pop_median_z=float(Z.loc[g_x, k].median()), cross_pop_frac_positive=float((Z.loc[g_x, k] > 0).mean()),
                             within_profile_mean_z=float(zin.loc[g_w].mean())))
            for g in g_x: gene_rows.append(dict(population=k, marker_set=ct, gene=g, cross_pop_z=float(Z.loc[g, k])))
    R = pd.DataFrame(rows); G = pd.DataFrame(gene_rows)
    R.to_csv(f"{TAB}/l5_audit_extmarkers_scores.tsv", sep="\t", index=False); G.to_csv(f"{TAB}/l5_audit_extmarkers_gene_z.tsv", sep="\t", index=False)
    summ = []
    for k in L5_POPS:
        r = R[R.population == k].set_index("marker_set")
        c = r.cross_pop_mean_z; w = r.within_profile_mean_z
        summ.append(dict(population=k, literature_mapping=LIT[k], mapping_used=prof[k]["model_col"], excluded=k in EXCL, exclude_reason=EXCL.get(k, ""),
                         cross_z_L5IT=c["L5 IT"], cross_z_L5ET=c["L5 ET"], cross_z_NP=c["L5/6 NP"], cross_best=c.idxmax(), cross_ET_minus_IT=c["L5 ET"] - c["L5 IT"],
                         cross_margin_best_vs_second=float(c.max() - c.drop(c.idxmax()).max()),
                         within_best=w.idxmax(), within_ET_minus_IT=w["L5 ET"] - w["L5 IT"]))
    S = pd.DataFrame(summ)
    # Primary decision is the ET-vs-IT contrast (the question Prashant raised). No external population is L5/6 NP, so the
    # 3-way argmax is only a flag: an NP-high score in a whole-cortex pulldown is reported, not used for assignment.
    MARGIN = 0.1
    def et_it_call(d): return "L5 ET" if d > MARGIN else ("L5 IT" if d < -MARGIN else "indeterminate (|ET-IT| <= %.1f)" % MARGIN)
    S["cross_ET_vs_IT_call"] = S.cross_ET_minus_IT.map(et_it_call); S["within_ET_vs_IT_call"] = S.within_ET_minus_IT.map(et_it_call)
    def agree(r):
        if r.literature_mapping.startswith("mixed"): return "consistent (near zero)" if abs(r.cross_ET_minus_IT) <= MARGIN else "check"
        if r.literature_mapping.startswith("L4"): return "n/a (excluded)"
        return "yes" if r.cross_ET_vs_IT_call == r.literature_mapping else "NO"
    S["cross_agrees_with_literature"] = S.apply(agree, axis=1)
    S["np_flag"] = S.apply(lambda r: "NP set highest" if r.cross_best == "L5/6 NP" else "", axis=1)
    S.to_csv(f"{TAB}/l5_audit_extmarkers_summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 300)
    print("\n== cross-population mean z of Yao 2021 marker sets ==")
    print(R.pivot_table(index="population", columns="marker_set", values="cross_pop_mean_z").loc[L5_POPS, L5].round(2).to_string())
    print("\n== within-profile mean z (script-17 style, for comparison) ==")
    print(R.pivot_table(index="population", columns="marker_set", values="within_profile_mean_z").loc[L5_POPS, L5].round(2).to_string())
    print("\n== summary ==\n", S[["population", "literature_mapping", "mapping_used", "cross_ET_minus_IT", "cross_ET_vs_IT_call", "cross_agrees_with_literature", "within_ET_minus_IT", "within_ET_vs_IT_call", "cross_best", "np_flag"]].round(2).to_string())
    # figure: gene-level cross-population z heatmap (markers grouped by set) + set-score bars
    order = [g for ct in L5 for g in markers[ct].gene if g in Z.index]; sets = [ct for ct in L5 for g in markers[ct].gene if g in Z.index]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.16 * len(order) + 2.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]; H = Z.loc[order, L5_POPS]
    im = ax.imshow(H.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6); ax.set_xticks(range(len(L5_POPS))); ax.set_xticklabels([f"{p}{' (excl.)' if p in EXCL else ''}\nlit: {LIT[p]}" for p in L5_POPS], rotation=45, ha="right", fontsize=7)
    b = 0
    for ct in L5:
        n = sets.count(ct)
        if n: ax.axhline(b + n - 0.5, color="k", lw=0.8); ax.text(len(L5_POPS) - 0.4, b + n / 2, ct, va="center", ha="left", fontsize=8, fontweight="bold"); b += n
    ax.set_title(f"Yao 2021 MOp subclass markers: z across {len(E.columns)} external glutamatergic TRAP profiles", fontsize=9); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.12, label="cross-population z")
    ax = axes[1]; P = R.pivot_table(index="population", columns="marker_set", values="cross_pop_mean_z").loc[L5_POPS, L5]
    y = np.arange(len(L5_POPS)); cols = {"L5 IT": "#0072B2", "L5 ET": "#D55E00", "L5/6 NP": "#7F7F7F"}
    for i, ct in enumerate(L5): ax.barh(y + (i - 1) * 0.27, P[ct].values, 0.25, color=cols[ct], label=ct)
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y); ax.set_yticklabels(L5_POPS); ax.invert_yaxis(); ax.set_xlabel("mean cross-population z of marker set"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("set scores (decision = sign of ET minus IT; NP shown as a flag)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/L5_audit_external_markers.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{FIG}/L5_audit_external_markers.pdf", bbox_inches="tight")
    print("saved figure")


if __name__ == "__main__":
    main()
