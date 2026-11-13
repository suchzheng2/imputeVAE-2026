#!/usr/bin/env python3
"""Extension: H2 (profile improvement) and H3 (RiboTag abundance vs predicted delta) on newer RNA-seq RiboTag/TRAP datasets,
cross-cell-type specificity, and H2 for baseline methods. Statistics are imported from 08_profile_improvement.py unchanged."""
import argparse, json, os, re, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
s08 = import_module("08_profile_improvement")
TASK = s08.TASK; PROJECT = s08.PROJECT; PB = s08.PB; EXT = f"{TASK}/outputs/extension"; TAB = f"{EXT}/tables"; SEED = 42
POOLS = {"POOL_excitatory": ['L2/3 IT', 'L5 ET', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6 IT', 'L6b'], "POOL_htr3a": ['Vip', 'Lamp5', 'Sncg'], "POOL_L5_IT_ET": ['L5 IT', 'L5 ET']}

def add_pools(df):
    df = df.copy()
    for k, v in POOLS.items(): df[k] = df[v].mean(axis=1)
    return df

def load_model(tag="all", method=None):
    d = {k: add_pools(pd.read_csv(f"{PB}/{k}_pseudobulk_by_ct_{tag}.csv", index_col=0)) for k in ("nucleus", "imputed", "cell")}
    if method: d["imputed"] = add_pools(pd.read_csv(f"{EXT}/model_pseudobulk/imputed_pseudobulk_by_ct__{method}.csv", index_col=0))
    cols = [c for c in d["nucleus"].columns if c in d["imputed"].columns and c in d["cell"].columns]   # drops cell-only 'L6 IT Car3'
    d = {k: v[cols] for k, v in d.items()}
    d["pred"] = d["imputed"] - d["nucleus"]; d["gt"] = d["cell"] - d["nucleus"]
    return d

def populations(include_excluded=False):
    """list of dict(dataset, population, batch, model_col, reps DataFrame (genes x replicates, log scale), thr)"""
    out = []
    P = pd.read_csv(f"{EXT}/metadata/extension_populations.tsv", sep="\t")
    if "exclude" in P.columns and not include_excluded: P = P[~P.exclude.astype(bool)]   # Scnn1a_L4: no L4 in motor cortex (mapping audit 2026-09-24)
    for _, r in P.iterrows():
        out.append(dict(dataset=r.dataset, population=r.population, batch=r.batch, model_col=r.model_col, cell_class=r.cell_class, year=int(r.year),
                        reps=pd.read_csv(f"{EXT}/normalized/{r.file}", sep="\t", index_col=0), thr=1.0 if "RPKM" in r.expressed_rule else 0.0))
    for ct, b in s08.external_gse263082().items():
        for bn, df in b.items(): out.append(dict(dataset="GSE263082", population=ct, batch=bn, model_col=ct, cell_class="inhibitory", year=2024, reps=df, thr=0.0))
    x = pd.read_excel(f"{PROJECT}/data/external/furlanis2019/GSE77243_Greenberg_SupplementaryTable1.xlsx", sheet_name="allgenes.csv").set_index("Genenames"); x = x[~x.index.duplicated()]
    for ct, pre in (("Pvalb", "Pv"), ("Sst", "Sst"), ("Vip", "Vip")):
        reps = [c for c in x.columns if re.match(rf"^{pre}_Std_B\d+$", c)]
        out.append(dict(dataset="GSE77243", population=ct, batch="std", model_col=ct, cell_class="inhibitory", year=2016, reps=np.log2(x[reps].astype(float) + 1), thr=0.0))
    # META entries (equal-weight mean of batch means on common expressed genes)
    keys = {}
    for p in out: keys.setdefault((p["dataset"], p["population"]), []).append(p)
    for (ds, pop), ps in keys.items():
        if len(ps) > 1:
            common = None
            for p in ps:
                mu = p["reps"].mean(axis=1); g = mu.index[mu > p["thr"]]; common = g if common is None else common.intersection(g)
            meta = pd.concat([p["reps"].loc[common].mean(axis=1).rename(p["batch"]) for p in ps], axis=1)
            out.append(dict(dataset=ds, population=pop, batch="META", model_col=ps[0]["model_col"], cell_class=ps[0]["cell_class"], year=ps[0]["year"], reps=meta, thr=ps[0]["thr"], is_meta=True))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_perm", type=int, default=1000); a = ap.parse_args(); nb = a.n_perm
    rng = np.random.default_rng(SEED); m = load_model(); pops = populations(); h2, h3 = [], []
    for p in pops:
        mu = p["reps"].mean(axis=1); genes = mu.index[mu > p["thr"]].intersection(m["nucleus"].index); ct = p["model_col"]
        st = s08.profile_stats(mu, m, ct, genes, rng, nb)
        if not p.get("is_meta") and p["reps"].shape[1] >= 2:
            lo, hi, mean_rep = s08.replicate_boot(p["reps"], m, ct, genes, rng, nb); st.update(replicate_boot_ci_lo=lo, replicate_boot_ci_hi=hi)
        st.update(dataset=p["dataset"], population=p["population"], batch=p["batch"], model_col=ct, cell_class=p["cell_class"], year=p["year"], n_replicates=p["reps"].shape[1], method="ImputeVAE"); h2.append(st)
        h = s08.h3(mu, m, ct, genes, rng, nb); h.update(dataset=p["dataset"], population=p["population"], batch=p["batch"], model_col=ct, cell_class=p["cell_class"], year=p["year"]); h3.append(h)
        print(f"{p['dataset']:10s} {p['population']:14s} {p['batch']:11s} -> {ct:16s} n={len(genes)} nuc {st['spearman_nucleus']:.3f} imp {st['spearman_imputed']:.3f} cell {st['spearman_realcell']:.3f} | H3 rho {h['spearman_ribotag_vs_pred_delta']:.3f} partial {h['partial_spearman_pred_given_nucexpr']:.3f}", flush=True)
    pd.DataFrame(h2).to_csv(f"{TAB}/extension_H2_profile_improvement.tsv", sep="\t", index=False)
    pd.DataFrame(h3).to_csv(f"{TAB}/extension_H3_translational_support.tsv", sep="\t", index=False)
    # ---------------- specificity: one profile per external population (META if available, else single batch)
    chosen = {}
    for p in pops:
        k = (p["dataset"], p["population"])
        if p.get("is_meta") or k not in chosen: chosen[k] = p
    common = None
    for p in chosen.values():
        mu = p["reps"].mean(axis=1); g = mu.index[mu > p["thr"]]; common = g if common is None else common.intersection(g)
    common = common.intersection(m["nucleus"].index)
    E = pd.concat([p["reps"].loc[common].mean(axis=1).rename(f"{k[0]}|{k[1]}") for k, p in chosen.items()], axis=1)
    cts = [c for c in m["nucleus"].columns if c not in ("ALL",)]
    spec = []
    for kind in ("nucleus", "imputed", "cell"):
        M = m[kind].loc[common, cts]
        for centered in (False, True):
            Mx = M.sub(M[[c for c in cts if not c.startswith("POOL")]].mean(axis=1), axis=0) if centered else M
            Ex = E.sub(E.mean(axis=1), axis=0) if centered else E
            for ecol in E.columns:
                k = tuple(ecol.split("|")); matched = chosen[k]["model_col"]
                rr = {c: stats.spearmanr(Mx[c], Ex[ecol])[0] for c in cts}
                base = [c for c in cts if not c.startswith("POOL")] + ([matched] if matched.startswith("POOL") else [])
                rank = 1 + sum(rr[c] > rr[matched] for c in base if c != matched)
                for c in cts: spec.append(dict(kind=kind, centered=centered, external=ecol, model_ct=c, spearman=rr[c], matched_ct=matched, is_matched=(c == matched), matched_rank=rank, n_candidates=len(base), n_genes=len(common)))
    pd.DataFrame(spec).to_csv(f"{TAB}/extension_specificity.tsv", sep="\t", index=False)
    s = pd.DataFrame(spec); s = s[s.is_matched]
    print("\nmatched-CT rank among model cell types (1 = best):"); print(s.pivot_table(index="external", columns=["centered", "kind"], values="matched_rank").to_string())
    # ---------------- baseline methods H2
    rows = []
    methods = [f[len("imputed_pseudobulk_by_ct__"):-4] for f in sorted(os.listdir(f"{EXT}/model_pseudobulk")) if f.startswith("imputed_pseudobulk_by_ct__")]
    for meth in methods:
        mm = load_model(method=meth)
        for p in pops:
            if not (p.get("is_meta") or sum(1 for q in pops if (q["dataset"], q["population"]) == (p["dataset"], p["population"])) == 1): continue
            mu = p["reps"].mean(axis=1); genes = mu.index[mu > p["thr"]].intersection(mm["nucleus"].index)
            st = s08.profile_stats(mu, mm, p["model_col"], genes, rng, max(200, nb // 5))
            st.update(dataset=p["dataset"], population=p["population"], batch=p["batch"], model_col=p["model_col"], cell_class=p["cell_class"], method=meth); rows.append(st)
        print(f"baseline {meth} done", flush=True)
    imp = pd.DataFrame(h2); imp = imp[imp.apply(lambda r: any((r.dataset, r.population, r.batch) == (x["dataset"], x["population"], x["batch"]) for x in rows), axis=1)]
    allm = pd.concat([imp, pd.DataFrame(rows)], ignore_index=True)
    allm.to_csv(f"{TAB}/extension_H2_method_comparison.tsv", sep="\t", index=False)
    print(allm.pivot_table(index=["dataset", "population"], columns="method", values="spearman_imputed").round(3).to_string())

if __name__ == "__main__":
    main()
