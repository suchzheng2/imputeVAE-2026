#!/usr/bin/env python3
"""Phase 3: harmonize gene identifiers to current mouse symbols for every dataset.

Rules (documented per dataset in outputs/gene_mapping/<GSE>_mapping_audit.tsv):
  * Ensembl IDs: strip version suffix, map with data/external/fractionation/ensembl_mouse_id2name.csv
    (BioMart export, 78,348 rows). Unmapped -> dropped.
  * Multi-symbol rows ("A // B", "A///B") -> dropped as ambiguous.
  * Duplicate symbols: raw counts are SUMMED; FPKM (GSE210071) uses the MEDIAN across rows.
  * GSE13379: probe -> symbol via GPL1261.annot; matrix kept at probe level, mapping table saved;
    gene-level collapse (median of log2 GCRMA across probes) happens in preprocessing.
Outputs: outputs/harmonized/<GSE>[_<batch>]_symbol_matrix.tsv.gz (symbol-indexed, original values),
         outputs/gene_mapping/<GSE>_mapping_audit.tsv, outputs/gene_mapping/mapping_summary.tsv
"""
import gzip
import os
import re

import numpy as np
import pandas as pd

PROJECT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
TASK = os.path.join(PROJECT, "validation/trap_ribotag_external_validation")
DATA = f"{TASK}/data"
MAP = f"{TASK}/outputs/gene_mapping"
HARM = f"{TASK}/outputs/harmonized"
os.makedirs(MAP, exist_ok=True); os.makedirs(HARM, exist_ok=True)

MODEL_GENES = set(pd.read_csv(f"{PROJECT}/data/training_gene_list_v3.csv", header=None)[0])
SHOWCASE = ["Malat1", "Xist", "Meg3", "Snhg11", "Neat1", "Kcnq1ot1", "Snap25", "Camk2n1", "Tmsb4x", "Pvalb", "Sst"]
ens = pd.read_csv(f"{PROJECT}/data/external/fractionation/ensembl_mouse_id2name.csv")
ENS2SYM = dict(zip(ens.iloc[:, 0].str.split(".").str[0], ens.iloc[:, 1]))
summary = []


def audit(name, n_in, n_mapped, n_ambig, n_dup_rows, final, extra=""):
    genes = set(final.index)
    rec = dict(dataset=name, input_rows=n_in, mapped_rows=n_mapped, ambiguous_dropped=n_ambig,
               duplicate_rows_collapsed=n_dup_rows, final_unique_genes=len(genes),
               overlap_model_10134=len(genes & MODEL_GENES),
               showcase_present=",".join(g for g in SHOWCASE if g in genes),
               showcase_absent=",".join(g for g in SHOWCASE if g not in genes), notes=extra)
    summary.append(rec)
    print(f"{name}: in={n_in} mapped={n_mapped} ambiguous={n_ambig} dup_rows={n_dup_rows} "
          f"final={len(genes)} overlap_model={rec['overlap_model_10134']}  absent={rec['showcase_absent']}")


def collapse(df, how):
    dup_rows = int(df.index.duplicated().sum())
    out = df.groupby(level=0).sum() if how == "sum" else df.groupby(level=0).median()
    return out, dup_rows


# ---------------- GSE210071 (symbols, FPKM)
df = pd.read_csv(f"{DATA}/GSE210071/GSE210071_FPKM_with_annotations.txt.gz", sep="\t", keep_default_na=False)
n_in = len(df)
df = df[df.Symbol != "NULL"]
n_mapped = len(df)
ambig = df.Symbol.str.contains(" // ")
n_amb = int(ambig.sum()); df = df[~ambig]
vals = df.set_index("Symbol").drop(columns=["FASTDB_ID", "Description", "Coordinates_mm10"]).astype(float)
vals, ndup = collapse(vals, "median")
vals.to_csv(f"{HARM}/GSE210071_symbol_matrix.tsv.gz", sep="\t")
audit("GSE210071", n_in, n_mapped, n_amb, ndup, vals, "FPKM; NULL symbols dropped; duplicates -> median")
pd.DataFrame({"rule": ["NULL symbol dropped", "' // ' multi-symbol dropped", "duplicate symbol -> median FPKM"],
              "n": [n_in - n_mapped, n_amb, ndup]}).to_csv(f"{MAP}/GSE210071_mapping_audit.tsv", sep="\t", index=False)

# ---------------- GSE121199 (Ensembl, counts)
df = pd.read_csv(f"{DATA}/GSE121199/GSE121199_all_counts.txt.gz", sep="\t", index_col=0)
n_in = len(df)
df.index = df.index.str.split(".").str[0]
sym = df.index.map(ENS2SYM)
mapped = sym.notna()
m = df[mapped].copy(); m.index = sym[mapped]
n_mapped = len(m)
m, ndup = collapse(m, "sum")
m.to_csv(f"{HARM}/GSE121199_symbol_matrix.tsv.gz", sep="\t")
audit("GSE121199", n_in, n_mapped, 0, ndup, m, "Ensembl->symbol via BioMart export; duplicates -> sum")
pd.DataFrame({"ensembl_id": df.index, "symbol": sym, "mapped": mapped}).to_csv(f"{MAP}/GSE121199_mapping_audit.tsv", sep="\t", index=False)

# ---------------- GSE263082 (5 files)
files = {"batch1": "GSE263082_NS046_raw-counts_gt40.txt.gz", "batch2": "GSE263082_Max_NS68_raw.txt.gz",
         "batch3": "GSE263082_Batch3_RawGeneCounts.txt.gz", "batch4": "GSE263082_Batch4_RawGeneCounts.txt.gz",
         "batch5": "GSE263082_Batch5_RawGeneCounts.txt.gz"}
agree_rows = []
for b, fn in files.items():
    df = pd.read_csv(f"{DATA}/GSE263082/{fn}", sep="\t")
    n_in = len(df)
    if b in ("batch1", "batch2"):
        df = df.set_index("ID"); n_mapped = n_in; n_amb = 0; extra = "symbols as supplied"
    else:
        df.columns = ["ensembl_id", "symbol"] + list(df.columns[2:])
        df["symbol"] = df["symbol"].astype(str).str.lstrip("'")
        df["ensembl_id"] = df["ensembl_id"].str.split(".").str[0]
        bm = df["ensembl_id"].map(ENS2SYM)
        both = bm.notna() & df["symbol"].ne("nan") & df["symbol"].ne("")
        agree = (bm[both] == df["symbol"][both]).mean()
        agree_rows.append((b, int(both.sum()), round(float(agree), 4)))
        # prefer embedded symbol; use the BioMart symbol when the embedded one is empty OR when only
        # the BioMart symbol is in the model gene set (rescues ~110 renamed genes, e.g. Sept7 -> Septin7)
        emb_in = df["symbol"].isin(MODEL_GENES); bm_in = bm.isin(MODEL_GENES)
        use_bm = df["symbol"].isin(["nan", ""]) | (~emb_in & bm_in)
        df["symbol"] = np.where(use_bm, bm.fillna(""), df["symbol"])
        extra_rescued = int((~emb_in & bm_in).sum())
        df = df[df["symbol"] != ""]
        n_mapped = len(df); n_amb = 0
        df = df.drop(columns=["ensembl_id"]).set_index("symbol")
        extra = f"embedded symbol (leading quote stripped); BioMart agreement {agree:.3f} on {int(both.sum())} rows; {extra_rescued} rows switched to BioMart symbol to match model gene set"
    df.columns = [str(c) for c in df.columns]
    df, ndup = collapse(df.astype(float), "sum")
    df.to_csv(f"{HARM}/GSE263082_{b}_symbol_matrix.tsv.gz", sep="\t")
    audit(f"GSE263082_{b}", n_in, n_mapped, n_amb, ndup, df, extra)
pd.DataFrame(agree_rows, columns=["batch", "rows_with_both", "embedded_vs_biomart_agreement"]).to_csv(
    f"{MAP}/GSE263082_mapping_audit.tsv", sep="\t", index=False)

# ---------------- GSE174303 (symbol + Entrez, counts)
df = pd.read_csv(f"{DATA}/GSE174303/GSE174303_counts_matrix.txt.gz", sep="\t")
n_in = len(df)
df = df.drop(columns=["Gene"]).set_index("GeneName")
df, ndup = collapse(df.astype(float), "sum")
df.to_csv(f"{HARM}/GSE174303_symbol_matrix.tsv.gz", sep="\t")
audit("GSE174303", n_in, n_in, 0, ndup, df, "symbols as supplied; duplicates -> sum")

# ---------------- GSE13379 (probes -> symbols)
with gzip.open(f"{DATA}/GSE13379/GPL1261.annot.gz", "rt") as f:
    lines = [l for l in f if not l.startswith(("!", "^", "#"))]
hdr = next(i for i, l in enumerate(lines) if l.startswith("ID\t"))
ann = pd.DataFrame([l.rstrip("\n").split("\t") for l in lines[hdr + 1:]], columns=lines[hdr].rstrip("\n").split("\t"))
ann = ann[["ID", "Gene symbol"]].rename(columns={"ID": "probe", "Gene symbol": "symbol"})
ann["symbol"] = ann["symbol"].fillna("")
ann["status"] = np.select([ann.symbol == "", ann.symbol.str.contains("///")], ["unannotated", "ambiguous"], "mapped")
ann.to_csv(f"{MAP}/GSE13379_probe_to_symbol.tsv", sep="\t", index=False)
# series matrix -> probe-level matrix
rows = []
with gzip.open(f"{DATA}/GSE13379/GSE13379_series_matrix.txt.gz", "rt") as f:
    for line in f:
        if line.startswith("!") or not line.strip():
            continue
        rows.append(line.rstrip("\n").replace('"', "").split("\t"))
sm = pd.DataFrame(rows[1:], columns=rows[0]).set_index("ID_REF").astype(float)
sm.to_csv(f"{HARM}/GSE13379_probe_matrix.tsv.gz", sep="\t")
ok = ann[ann.status == "mapped"].set_index("probe")["symbol"]
ok = ok[ok.index.isin(sm.index)]
final = pd.Index(ok.unique())
n_amb = int((ann.status == "ambiguous").sum())
audit("GSE13379", len(sm), int(len(ok)), n_amb, int(len(ok) - len(final)), pd.DataFrame(index=final),
      "probe matrix kept; gene-level collapse (median log2 across probes) deferred to preprocessing")

pd.DataFrame(summary).to_csv(f"{MAP}/mapping_summary.tsv", sep="\t", index=False)
print("\nwrote", f"{MAP}/mapping_summary.tsv")
