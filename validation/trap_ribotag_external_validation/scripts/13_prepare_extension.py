#!/usr/bin/env python3
"""Extension: metadata, symbol harmonization and normalization of the newer RNA-seq RiboTag/TRAP datasets.
Sample selection is explicit by GSM / column (wild-type, baseline/control, IP only). Counts -> TMM (extval_utils) + logCPM
(prior 2) within dataset; GSE211162 supplies RPKM only -> log2(RPKM+1). Expressed threshold: logCPM > 0 (CPM > 1) or RPKM > 1.
Outputs: outputs/extension/metadata/extension_samples.tsv, extension_populations.tsv, gene_mapping_audit.tsv,
         outputs/extension/normalized/<dataset>__<population>__<batch>.tsv.gz (genes x replicates), harmonized count matrices."""
import glob, gzip, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extval_utils import tmm_factors, log_cpm
TASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PROJECT = os.path.dirname(os.path.dirname(TASK))
DATA = f"{TASK}/data"; EXT = f"{TASK}/outputs/extension"
MODEL = set(pd.read_csv(f"{PROJECT}/data/training_gene_list_v3.csv", header=None)[0])
samples, pops, audit = [], [], []

def read_gsm(acc, gsm, symbol_col=None):
    f = glob.glob(f"{DATA}/{acc}/raw/{gsm}_*.gz"); assert len(f) == 1, (acc, gsm, f)
    if symbol_col:  # GSE172484: IDs, Symbols, count
        d = pd.read_csv(f[0], sep="\t"); s = d.groupby("Symbols")[d.columns[2]].sum()
    else:
        d = pd.read_csv(f[0], sep="\t", header=None, names=["g", "c"]); d = d[~d.g.str.startswith("__")]; s = d.groupby("g").c.sum()
    return s.rename(gsm)

def finish(acc, counts, groups, info, is_log=False, thr_note="logCPM>0"):
    """counts: genes x samples; groups: dict sample -> (population, batch). Writes normalized per population/batch."""
    n_in = len(counts)
    if not is_log:
        counts = counts.fillna(0); counts.to_csv(f"{EXT}/harmonized/{acc}_symbol_counts.tsv.gz", sep="\t")
        cpm = counts / counts.sum() * 1e6; min_n = min(pd.Series([g for g in groups.values()]).value_counts())
        keep = (cpm >= 1).sum(axis=1) >= min_n
        counts = counts[keep]; f = tmm_factors(counts); L = log_cpm(counts, f, prior=2.0)
        qc = pd.DataFrame({"sample": counts.columns, "lib_size": counts.sum().values, "tmm_factor": f.values, "detected_cpm_gt1": (cpm[keep] >= 1).sum().values})
    else:
        L = counts; qc = pd.DataFrame({"sample": L.columns, "lib_size": np.nan, "tmm_factor": np.nan, "detected_cpm_gt1": (L > 1).sum().values})
    qc["dataset"] = acc; qc.to_csv(f"{EXT}/normalized/{acc}_qc.tsv", sep="\t", index=False)
    sc = L.corr(method="spearman")
    for (pop, batch) in sorted(set(groups.values())):
        cols = [c for c, g in groups.items() if g == (pop, batch)]
        L[cols].to_csv(f"{EXT}/normalized/{acc}__{pop}__{batch}.tsv.gz", sep="\t")
        within = sc.loc[cols, cols].values[np.triu_indices(len(cols), 1)]
        pops.append(dict(dataset=acc, population=pop, batch=batch, n_reps=len(cols), median_within_spearman=float(np.median(within)), **info[pop],
                         file=f"{acc}__{pop}__{batch}.tsv.gz", expressed_rule=thr_note))
        for c in cols: samples.append(dict(dataset=acc, sample=c, population=pop, batch=batch, include_primary=True, **{k: info[pop][k] for k in ("tissue", "assay")}))
    audit.append(dict(dataset=acc, input_genes=n_in, genes_after_filter=len(L), overlap_model_10134=len(set(L.index) & MODEL)))
    print(f"{acc}: {n_in} -> {len(L)} genes, model overlap {len(set(L.index) & MODEL)}; groups {sorted(set(groups.values()))}")

# ---------------- GSE133291 Furlanis 2019 (neocortex only; K-suffixed columns are not in GEO metadata -> excluded)
ann = pd.read_csv(f"{DATA}/GSE210071/GSE210071_FPKM_with_annotations.txt.gz", sep="\t", usecols=["FASTDB_ID", "Symbol"], keep_default_na=False)
ann = ann[(ann.Symbol != "NULL") & ~ann.Symbol.str.contains(" // ")].set_index("FASTDB_ID").Symbol
fur = pd.read_csv(f"{PROJECT}/data/external/furlanis2019/GSE133291_selectedCounts.txt.gz", sep=" ", quotechar='"')   # space-delimited R write.table
fur.index = fur.index.astype(str).str.strip('"')
cortex = {f"CamK_{i}": "CamK2a_exc" for i in range(1, 5)} | {f"PV_{i}": "PV" for i in range(1, 5)} | {f"Scnn_{i}": "Scnn1a_L4" for i in range(1, 5)} | {f"SST_{i}": "SST" for i in range(1, 5)} | {f"VIP_W{i}": "VIP" for i in range(1, 5)}
assert all(c in fur.columns for c in cortex), [c for c in cortex if c not in fur.columns]
m = fur.loc[fur.index.intersection(ann.index), list(cortex)]; m.index = ann.loc[m.index].values; m = m.groupby(level=0).sum().round()
audit.append(dict(dataset="GSE133291_mapping", input_genes=len(fur), genes_after_filter=len(m), overlap_model_10134=len(set(m.index) & MODEL)))
info = {p: dict(tissue="neocortex", assay="RiboTag IP RNA-seq (constitutive-exon counts)", year=2019, model_col=mc, nuc_ct=nc, cell_class=cl, note=nt) for p, mc, nc, cl, nt in
        [("CamK2a_exc", "POOL_excitatory", "L5 IT", "excitatory", "pan-excitatory driver -> unweighted pool of 7 excitatory CTs"), ("Scnn1a_L4", "L5 IT", "L5 IT", "excitatory", "EXCLUDED 2026-09-24: L4 spiny stellate driver (Furlanis 2019); motor cortex has no L4 -> no counterpart"),
         ("PV", "Pvalb", "Pvalb", "inhibitory", ""), ("SST", "Sst", "Sst", "inhibitory", ""), ("VIP", "Vip", "Vip", "inhibitory", "")]}
finish("GSE133291", m, {c: (p, "cortex") for c, p in cortex.items()}, info)
for c in fur.columns:
    if c not in cortex and c not in ("length", "selectedConst"):
        samples.append(dict(dataset="GSE133291", sample=c, population=c.split("_")[0], batch="NA", include_primary=False, tissue="hippocampus or not in GEO metadata", assay="RiboTag IP", exclusion_reason="hippocampus (W) or K-suffixed column absent from GEO sample records"))

# ---------------- GSE172484 motor cortex L5b (Gprin3 lower, Colgalt2 upper); WT baseline + SOD- littermate controls
g = {"GSM5257804": ("Gprin3_L5b", "baselineM1"), "GSM5257805": ("Gprin3_L5b", "baselineM1"), "GSM5257809": ("Colgalt2_L5b", "baselineM1"), "GSM5257810": ("Colgalt2_L5b", "baselineM1"),
     "GSM5257813": ("Gprin3_L5b", "SODnegM1"), "GSM5257814": ("Gprin3_L5b", "SODnegM1"), "GSM5257818": ("Colgalt2_L5b", "SODnegCtx"), "GSM5257819": ("Colgalt2_L5b", "SODnegCtx"), "GSM5257820": ("Colgalt2_L5b", "SODnegCtx")}
m = pd.concat([read_gsm("GSE172484", k, symbol_col=True) for k in g], axis=1)
info = {p: dict(tissue="motor cortex (M1)" , assay="bacTRAP IP RNA-seq", year=2022, model_col="L5 ET", nuc_ct="L5 ET", cell_class="excitatory", note=n) for p, n in [("Gprin3_L5b", "lower L5b pyramidal (PT-like)"), ("Colgalt2_L5b", "upper L5b pyramidal (Colgalt2 = Glt25d2); SODneg batch is whole cortex")]}
finish("GSE172484", m, g, info)
# ---------------- GSE202172 S100a10 L5a, vehicle group-housed IP
g = {k: ("S100a10_L5a", "vehGH") for k in ["GSM6102619", "GSM6102620", "GSM6102627", "GSM6102628"]}
finish("GSE202172", pd.concat([read_gsm("GSE202172", k) for k in g], axis=1), g, {"S100a10_L5a": dict(tissue="cortex", assay="bacTRAP IP RNA-seq", year=2022, model_col="L5 IT", nuc_ct="L5 IT", cell_class="excitatory", note="L5a corticostriatal -> L5 IT")})
# ---------------- GSE141337 Glt25d2 L5b saline IP
g = {k: ("Glt25d2_L5b", "saline") for k in ["GSM4201273", "GSM4201274", "GSM4201275"]}
finish("GSE141337", pd.concat([read_gsm("GSE141337", k) for k in g], axis=1), g, {"Glt25d2_L5b": dict(tissue="cerebral cortex", assay="bacTRAP IP RNA-seq", year=2019, model_col="L5 ET", nuc_ct="L5 ET", cell_class="excitatory", note="RNA-seq successor of the 2008 DU9 line")})
# ---------------- GSE142028 Rbp4 cortex IP
g = {k: ("Rbp4_L5", "cortex") for k in ["GSM4217838", "GSM4217840"]}
finish("GSE142028", pd.concat([read_gsm("GSE142028", k) for k in g], axis=1), g, {"Rbp4_L5": dict(tissue="cerebral cortex", assay="Cre-dependent TRAP IP RNA-seq", year=2019, model_col="POOL_L5_IT_ET", nuc_ct="L5 IT", cell_class="excitatory", note="Rbp4-Cre KL100 = pan-L5 IT + PT/ET, not NP (Gerfen 2013; Tasic 2018) -> unweighted pool of L5 IT and L5 ET (changed from L5 IT on 2026-09-24)")})
# ---------------- GSE141874 Htr3a interneurons
g = {k: ("Htr3a_IN", "cortex") for k in ["GSM4215271", "GSM4215272", "GSM4215273"]}
finish("GSE141874", pd.concat([read_gsm("GSE141874", k) for k in g], axis=1), g, {"Htr3a_IN": dict(tissue="cerebral cortex", assay="Cre-dependent TRAP IP RNA-seq", year=2019, model_col="POOL_htr3a", nuc_ct="Vip", cell_class="inhibitory", note="Htr3a+ = Vip + Lamp5 + Sncg (unweighted pool)")})
# ---------------- GSE139684 Aldh1l1 astrocytes, PFC controls
g = {k: ("Aldh1l1_astro", "PFCctrl") for k in ["GSM4146176", "GSM4146177", "GSM4146178"]}
finish("GSE139684", pd.concat([read_gsm("GSE139684", k) for k in g], axis=1), g, {"Aldh1l1_astro": dict(tissue="prefrontal cortex", assay="bacTRAP IP RNA-seq", year=2019, model_col="Astro", nuc_ct="Astro", cell_class="glia", note="unstressed controls only")})
# ---------------- GSE99791 GFAP RiboTag astrocytes, 4 months, motor + visual cortex IP
with gzip.open(f"{DATA}/GSE99791/GSE99791_raw.txt.gz", "rt") as fh: hdr = fh.readline().rstrip("\n").split("\t")
d = pd.read_csv(f"{DATA}/GSE99791/GSE99791_raw.txt.gz", sep="\t", skiprows=1, header=None, names=["id"] + hdr[1:])
d["symbol"] = d["Annotation/Divergence"].astype(str).str.split("|").str[0]
cols = [f"4mo_MC_rep{i}" for i in (1, 2, 3)] + [f"4mo_VC_rep{i}" for i in (1, 2, 3)]
d["tot"] = d[cols].sum(axis=1); d = d.sort_values("tot", ascending=False).drop_duplicates("symbol").set_index("symbol")   # one RefSeq row per gene: highest total
g = {c: ("GFAP_astro", "MC" if "_MC_" in c else "VC") for c in cols}
finish("GSE99791", d[cols].round(), g, {"GFAP_astro": dict(tissue="motor cortex (MC) / visual cortex (VC)", assay="RiboTag IP RNA-seq (HOMER counts)", year=2017, model_col="Astro", nuc_ct="Astro", cell_class="glia", note="4-month adults; aged mice excluded")})
# ---------------- GSE211162 mature oligodendrocyte TRAP, frontal cortex (RPKM only)
d = pd.read_csv(f"{DATA}/GSE211162/GSE211162_Oligodendrocyte_TRAP-seq_RPKM.csv.gz", index_col=0)
cols = [c for c in d.columns if c.startswith("cortex_")]; d = d[cols].groupby(level=0).median(); d = d[(d > 1).sum(axis=1) >= 4]
finish("GSE211162", np.log2(d + 1), {c: ("Oligo_TRAP", "frontalCtx") for c in cols}, {"Oligo_TRAP": dict(tissue="frontal cortex", assay="TRAP IP RNA-seq (RPKM only)", year=2023, model_col="Oligo", nuc_ct="Oligo", cell_class="glia", note="no raw counts supplied; log2(RPKM+1); expressed = RPKM>1")}, is_log=True, thr_note="log2(RPKM+1)>1")

pd.DataFrame(samples).to_csv(f"{EXT}/metadata/extension_samples.tsv", sep="\t", index=False)
POPS = pd.DataFrame(pops)
EXCLUDE = {"Scnn1a_L4": "L4 driver; motor cortex has no L4 (Prashant, 2026-09-24)",
           "Rbp4_L5": "pan-L5 driver, mixed IT+ET (Gerfen 2013; Tasic 2018); no single matching subclass -> skipped rather than pooled (2026-09-25)"}
POPS["exclude"] = POPS.population.isin(EXCLUDE); POPS["exclude_reason"] = POPS.population.map(EXCLUDE).fillna("")
POPS["mapping_evidence"] = "see outputs/extension/metadata/l5_mapping_evidence.md"
POPS.to_csv(f"{EXT}/metadata/extension_populations.tsv", sep="\t", index=False)
pd.DataFrame(audit).to_csv(f"{EXT}/metadata/gene_mapping_audit.tsv", sep="\t", index=False)
print(pd.DataFrame(pops)[["dataset", "population", "batch", "n_reps", "median_within_spearman", "model_col"]].to_string(index=False))
