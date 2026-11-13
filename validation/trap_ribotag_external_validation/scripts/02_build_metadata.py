#!/usr/bin/env python3
"""Phase 2: explicit sample metadata for all five GEO datasets.

Labels are derived from GEO series-matrix sample records (title, characteristics,
description) and reconciled against the expression-matrix column names.
Fails loudly if a primary-analysis sample has no matrix column or a matrix column
has no metadata. Writes one TSV per dataset and outputs/metadata/all_samples.tsv.

Run from project root in the impute2 env:
    python validation/trap_ribotag_external_validation/scripts/02_build_metadata.py
"""
import gzip
import os
import re
import sys

import pandas as pd

TASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(TASK, "data")
OUT = os.path.join(TASK, "outputs", "metadata")
os.makedirs(OUT, exist_ok=True)

COLS = ["gsm", "column", "dataset", "species", "tissue", "cell_type", "assay",
        "compartment", "genotype", "treatment", "sex", "replicate", "batch",
        "source_file", "platform", "include_primary", "exclusion_reason", "notes"]


def read_series_matrix(path):
    rows = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("!Sample_"):
                k, *v = line.rstrip("\n").split("\t")
                rows.setdefault(k, []).append([x.strip('"') for x in v])
    return rows


def matrix_columns(path, sep="\t", skip=0):
    with gzip.open(path, "rt") as f:
        for _ in range(skip):
            f.readline()
        return f.readline().rstrip("\n").split(sep)


def check(df, cols, dataset, id_col="column"):
    """Every primary sample has a column; report matrix columns without metadata."""
    have = set(df[id_col].dropna())
    prim = df[df.include_primary]
    missing = prim[~prim[id_col].isin(cols)]
    if len(missing):
        sys.exit(f"[{dataset}] primary samples without matrix column: {missing[id_col].tolist()}")
    orphan = [c for c in cols if c not in have]
    return orphan


# ---------------------------------------------------------------- GSE210071
def gse210071():
    sm = read_series_matrix(f"{DATA}/GSE210071/GSE210071-GPL17021_series_matrix.txt.gz")
    sm2 = read_series_matrix(f"{DATA}/GSE210071/GSE210071-GPL24247_series_matrix.txt.gz")
    cols = matrix_columns(f"{DATA}/GSE210071/GSE210071_FPKM_with_annotations.txt.gz")
    recs = []
    # GPL17021: 18 compartment samples. Title: "Primary cortical cells, <Comp>, Triptolide-3hrs, <stim>, repN"
    # Matrix columns: S{1,2,4}_{N,C,W}_{Ctl,Bic}. Culture IDs S1/S2/S4 <-> rep1/rep2/rep3 (in that order).
    culture = {"1": "S1", "2": "S2", "3": "S4"}
    comp_code = {"Cytosol": "C", "Nucleus": "N", "Whole Cell Extract": "W"}
    stim_code = {"unstimulated": "Ctl", "Bicuculline-1hr": "Bic"}
    for t, g in zip(sm["!Sample_title"][0], sm["!Sample_geo_accession"][0]):
        m = re.match(r"Primary cortical cells, (.+?), Triptolide-3hrs, (.+?), rep(\d)", t)
        comp, stim, rep = m.groups()
        col = f"{culture[rep]}_{comp_code[comp]}_{stim_code[stim]}"
        recs.append(dict(gsm=g, column=col, dataset="GSE210071", species="mouse",
                         tissue="primary cortical culture (embryonic, mixed neurons+glia)",
                         cell_type="mixed cortical culture", assay="RNA-seq FPKM",
                         compartment={"C": "cytosol", "N": "nucleus", "W": "whole_cell"}[comp_code[comp]],
                         genotype="wild type", treatment=f"triptolide 3h + {stim}",
                         sex="NA", replicate=rep, batch=f"culture {culture[rep]}",
                         source_file="GSE210071_FPKM_with_annotations.txt.gz", platform="GPL17021",
                         include_primary=(stim == "unstimulated"),
                         exclusion_reason="" if stim == "unstimulated" else "bicuculline: sensitivity only",
                         notes="ALL compartment samples received triptolide (transcription inhibitor) for 3 h; culture ID pairs compartments"))
    # GPL24247: 9 DRB samples, no compartment split. Columns Ctrl_/BicS_/BDNF_ N7/N8/N10
    stim_col = {"unstimulated": "Ctrl", "Bicuculline-1hr": "BicS", "BDNF-1hr": "BDNF"}
    cult = {"1": "N7", "2": "N8", "3": "N10"}
    for t, g in zip(sm2["!Sample_title"][0], sm2["!Sample_geo_accession"][0]):
        m = re.match(r"Primary cortical cells, DRB-2hrs, (.+?), rep(\d)", t)
        stim, rep = m.groups()
        col = f"{stim_col[stim]}_{cult[rep]}"
        recs.append(dict(gsm=g, column=col, dataset="GSE210071", species="mouse",
                         tissue="primary cortical culture (embryonic, mixed neurons+glia)",
                         cell_type="mixed cortical culture", assay="RNA-seq FPKM",
                         compartment="total (no fractionation)", genotype="wild type",
                         treatment=f"DRB 2h + {stim}", sex="NA", replicate=rep, batch=f"culture {cult[rep]}",
                         source_file="GSE210071_FPKM_with_annotations.txt.gz", platform="GPL24247",
                         include_primary=False, exclusion_reason="DRB series has no compartment split; different platform",
                         notes=""))
    df = pd.DataFrame(recs, columns=COLS)
    orphan = check(df, cols, "GSE210071")
    orphan = [c for c in orphan if c not in ("FASTDB_ID", "Symbol", "Description", "Coordinates_mm10")]
    assert not orphan, f"GSE210071 orphan columns: {orphan}"
    return df


# ---------------------------------------------------------------- GSE121199
def gse121199():
    sm = read_series_matrix(f"{DATA}/GSE121199/GSE121199_series_matrix.txt.gz")
    cols = matrix_columns(f"{DATA}/GSE121199/GSE121199_all_counts.txt.gz")[1:]
    titles = sm["!Sample_title"][0]
    gsms = sm["!Sample_geo_accession"][0]
    # Matrix columns: wc_/nuclear_/ribo_ + D1/D2 + M/F + rep. GEO titles list WC D1 rep1-8 and
    # D2 rep1-8, but the matrix has D1 x8 + wc_D1_M_1_2 (9) and D2 x6 + wc_D2_M_3_L8 (7), which
    # matches the GEO summary (D1 n=9, D2 n=7). Titles are therefore not a reliable 1:1 map for
    # the WC block; we map WC columns by (assay, subtype, sex, replicate) parsed from the column
    # name and assign GSMs in series order within each (assay, subtype) block, flagging the two
    # extra columns explicitly.
    recs = []
    used = set()

    def gsm_for(assay_word, sub, sex, rep):
        for t, g in zip(titles, gsms):
            if g in used:
                continue
            if t.startswith(assay_word) and f"{sub}-MSN" in t and f", {sex}," in t and t.endswith(f"Rep {rep}"):
                used.add(g)
                return g
        return ""

    for c in cols:
        m = re.match(r"(wc|nuclear|ribo)_(D[12])_([MF])_(\d+)(.*)", c)
        assay, sub, sex, rep, suffix = m.groups()
        assay_word = {"wc": "Whole cell", "nuclear": "Nuclear", "ribo": "RiboTag"}[assay]
        comp = {"wc": "whole_cell", "nuclear": "nucleus", "ribo": "ribosome_bound"}[assay]
        note, incl, excl, gsm = "", True, "", ""
        if suffix == "_L8":
            note = "technical re-sequencing (lane 8) of wc_D2_M_3 (Spearman 0.995): SUM counts into wc_D2_M_3"
            incl, excl = False, "technical replicate; merged into wc_D2_M_3"
        elif suffix == "_2":
            note = "second library of D1 rep1 (Spearman 0.983 with wc_D1_M_1; D1 markers confirm D1): excluded from primary, sensitivity only"
            incl, excl = False, "possible technical duplicate of wc_D1_M_1"
        else:
            gsm = gsm_for(assay_word, sub, {"M": "Male", "F": "Female"}[sex], rep)
        if assay == "ribo" and sex == "F":
            incl, excl = False, "female RiboTag; nuclear/whole-cell samples are male only"
        recs.append(dict(gsm=gsm, column=c, dataset="GSE121199", species="mouse",
                         tissue="nucleus accumbens", cell_type=f"{sub} MSN", assay={"wc": "whole-cell FACS RNA-seq", "nuclear": "nuclear FACS RNA-seq", "ribo": "RiboTag IP RNA-seq"}[assay],
                         compartment=comp, genotype=f"{sub}-Cre reporter / RiboTag", treatment="none",
                         sex={"M": "male", "F": "female"}[sex], replicate=rep, batch="NA",
                         source_file="GSE121199_all_counts.txt.gz", platform="GPL21103",
                         include_primary=incl, exclusion_reason=excl, notes=note))
    df = pd.DataFrame(recs, columns=COLS)
    unmapped_gsm = [g for g in gsms if g not in used]
    df.loc[df.column == "wc_D1_M_1_2", "notes"] += f"; GEO GSMs not assigned to any column: {unmapped_gsm}"
    check(df, cols, "GSE121199")
    return df


# ---------------------------------------------------------------- GSE263082
def gse263082():
    sm = read_series_matrix(f"{DATA}/GSE263082/GSE263082_series_matrix.txt.gz")
    titles, gsms = sm["!Sample_title"][0], sm["!Sample_geo_accession"][0]
    desc_col, desc_file = sm["!Sample_description"][0], sm["!Sample_description"][1]
    ch = sm["!Sample_characteristics_ch1"]
    files = {"NS046_raw-counts_gt40.txt": "GSE263082_NS046_raw-counts_gt40.txt.gz",
             "Max_NS68_raw.txt": "GSE263082_Max_NS68_raw.txt.gz",
             "Batch3_RawGeneCounts.txt": "GSE263082_Batch3_RawGeneCounts.txt.gz",
             "Batch4_RawGeneCounts.txt": "GSE263082_Batch4_RawGeneCounts.txt.gz",
             "Batch5_RawGeneCounts.txt": "GSE263082_Batch5_RawGeneCounts.txt.gz"}
    file_cols = {}
    for k, v in files.items():
        hdr = matrix_columns(f"{DATA}/GSE263082/{v}")
        file_cols[k] = [c for c in hdr if c not in ("ID", "", "'Symbol")]
    recs = []
    for i, (t, g) in enumerate(zip(titles, gsms)):
        chars = {c[i].split(": ")[0]: c[i].split(": ", 1)[1] for c in ch}
        batch = t.split("_")[0]
        geno = chars["genotype"]
        ct = "Pvalb" if chars["cell type"].startswith("PV") else "Sst"
        trt = chars["treatment"]
        wt = "Nr4a1" not in geno
        baseline = trt in ("Saline", "Homecage")
        incl = wt and baseline
        excl = "" if incl else ("Nr4a1 conditional knockout" if not wt else f"{trt}: stimulated, sensitivity only")
        recs.append(dict(gsm=g, column=desc_col[i], dataset="GSE263082", species="mouse",
                         tissue="cortex + hippocampus (pooled)", cell_type=ct, assay="RiboTag IP RNA-seq (IP only, no input)",
                         compartment="ribosome_bound", genotype=geno, treatment=trt, sex="NA",
                         replicate=t.split("_")[-1], batch=batch, source_file=files[desc_file[i]],
                         platform="GPL30172", include_primary=incl, exclusion_reason=excl,
                         notes="column ID from !Sample_description; NS68 file is pre-filtered to 11,808 genes"))
    df = pd.DataFrame(recs, columns=COLS)
    for k, v in files.items():
        sub = df[df.source_file == v]
        orphan = [c for c in file_cols[k] if c not in set(sub.column)]
        miss = [c for c in sub.column if c not in file_cols[k]]
        assert not orphan and not miss, f"GSE263082 {k}: orphan={orphan} missing={miss}"
    return df


# ---------------------------------------------------------------- GSE174303
def gse174303():
    sm = read_series_matrix(f"{DATA}/GSE174303/GSE174303_series_matrix.txt.gz")
    titles, gsms = sm["!Sample_title"][0], sm["!Sample_geo_accession"][0]
    cols = matrix_columns(f"{DATA}/GSE174303/GSE174303_counts_matrix.txt.gz")[2:]
    # titles: {CB,NP}_{CreNeg,KO,WT}_{RNAseq,TRAP}_R#   matrix: {CB,ML}_{...}_{input,TRAP}_R#
    t2c = {t: t.replace("NP_", "ML_").replace("_RNAseq_", "_input_") for t in titles}
    recs = []
    for t, g in zip(titles, gsms):
        reg, geno, assay, rep = t.split("_")
        col = t2c[t]
        recs.append(dict(gsm=g, column=col, dataset="GSE174303", species="mouse",
                         tissue={"CB": "CA1 cell-body layer (stratum pyramidale)", "NP": "CA1 neuropil / molecular layer"}[reg],
                         cell_type="CA1 pyramidal (Camk2a-Cre TRAP)", assay={"RNAseq": "input RNA-seq", "TRAP": "TRAP IP RNA-seq"}[assay],
                         compartment={"RNAseq": "total_input", "TRAP": "ribosome_bound"}[assay],
                         genotype={"WT": "Fmr1 WT; TRAP+", "KO": "Fmr1 KO; TRAP+", "CreNeg": "Cre-negative control"}[geno],
                         treatment="none", sex="NA", replicate=rep, batch=rep,
                         source_file="GSE174303_counts_matrix.txt.gz", platform="GPL13112",
                         include_primary=(geno == "WT"), exclusion_reason="" if geno == "WT" else f"{geno}: not primary",
                         notes="matrix uses ML/input where GEO titles use NP/RNAseq"))
    df = pd.DataFrame(recs, columns=COLS)
    orphan = check(df, cols, "GSE174303")
    assert not orphan, f"GSE174303 orphan columns: {orphan}"
    return df


# ---------------------------------------------------------------- GSE13379
def gse13379():
    sm = read_series_matrix(f"{DATA}/GSE13379/GSE13379_series_matrix.txt.gz")
    titles, gsms = sm["!Sample_title"][0], sm["!Sample_geo_accession"][0]
    ch = sm["!Sample_characteristics_ch1"]
    recs = []
    for i, (t, g) in enumerate(zip(titles, gsms)):
        frac = "TRAP" if ch[0][i] == "TRAP RNA" else "unbound"
        m = re.search(r"Driver: (\w+), Line: (\w+), Region: ([\w ]+)", ch[1][i]) if len(ch) > 1 else None
        driver, line, region = m.groups() if m else ("NA", "NA", "NA")
        pop = t.split(",")[0].strip()
        cortex = region == "Cortex"
        recs.append(dict(gsm=g, column=g, dataset="GSE13379", species="mouse", tissue=region,
                         cell_type=f"{pop} ({driver} / {line})", assay="Affymetrix 430 2.0 GCRMA (linear scale)",
                         compartment={"TRAP": "ribosome_bound", "unbound": "unbound_polysomal_supernatant"}[frac],
                         genotype=f"bacTRAP {driver} line {line}", treatment="none", sex="NA",
                         replicate=t.split()[-1], batch=line, source_file="GSE13379_series_matrix.txt.gz",
                         platform="GPL1261", include_primary=cortex and driver != "NA",
                         exclusion_reason="" if cortex else "non-cortical region or unlabeled control",
                         notes="GCRMA within replicate groups; values are NOT log-transformed"))
    df = pd.DataFrame(recs, columns=COLS)
    return df


if __name__ == "__main__":
    dfs = []
    for fn in (gse210071, gse121199, gse263082, gse174303, gse13379):
        df = fn()
        name = fn.__name__.upper()
        df.to_csv(f"{OUT}/{name}_samples.tsv", sep="\t", index=False)
        dfs.append(df)
        n = len(df)
        print(f"{name}: {n} samples, {int(df.include_primary.sum())} primary")
        print(df[df.include_primary].groupby(["cell_type", "compartment", "treatment"]).size().to_string(), "\n")
    pd.concat(dfs).to_csv(f"{OUT}/all_samples.tsv", sep="\t", index=False)
    print("wrote", f"{OUT}/all_samples.tsv")
