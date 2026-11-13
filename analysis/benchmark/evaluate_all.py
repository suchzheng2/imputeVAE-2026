"""
Split-aware benchmark: unified evaluation of every method on the held-out partition.

Every method directory under --base_dir holds ``imputed_eval.npy`` (log1p scale,
rows = eval nuclei in cache order).  Reference data (eval cells, eval nuclei,
training cells) come from cache/benchmark so no method can ship its own truth.

Metric families (rationale in notes/benchmark_review_2026-10-09.md):

  A. Cell-type-level mean accuracy   reference = eval-partition cells
       gene_corr, r2, gene_mse, profile_corr, shift, delta (pearson/spearman/r2/sign/auroc)
       CT-mean(train) is the sampling-noise CEILING for this family, not a trivial 1.0.
  B. Within-cell-type distribution    reference = eval cells of the same type
       variance_ratio, std_ratio, wasserstein (per gene, within CT), qq_corr (within CT),
       energy_distance and two-sample classifier AUROC in train-cell PCA space,
       with a real-vs-real noise floor.
  C. Per-cell information retention  reference = the input nucleus
       within_ct_spearman (does the output keep per-nucleus ordering?), variance_rank_spearman.
  D. Downstream utility              classifier trained on training cells, DE marker overlap,
       gene-gene coexpression (pooled and within the largest cell type).
  E. Compartment biology             ribosomal / cytoplasmic / nuclear-lncRNA delta signs,
       neuron-vs-glia enrichment ratio.
  Bootstrap CIs (resampling eval nuclei and eval cells within cell type) for the headline numbers.

Usage:
    python -m analysis.benchmark.evaluate_all --eval_partition test \
        --base_dir run/benchmark/test --output_dir run/benchmark/test/results
"""

import os
import re
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.benchmark.data_split import load_partition, load_gene_names, CACHE_DIR, MANIFEST_PATH

# ----------------------------------------------------------------------------
# Prespecified gene sets (same lists as analysis/eval_splitaware.py)
# ----------------------------------------------------------------------------
RIBOSOMAL_GENES = [
    'Rpl3', 'Rpl4', 'Rpl5', 'Rpl6', 'Rpl7', 'Rpl7a', 'Rpl8', 'Rpl9', 'Rpl10', 'Rpl10a', 'Rpl11', 'Rpl12',
    'Rpl13', 'Rpl13a', 'Rpl14', 'Rpl15', 'Rpl17', 'Rpl18', 'Rpl18a', 'Rpl19', 'Rpl21', 'Rpl22', 'Rpl23',
    'Rpl23a', 'Rpl24', 'Rpl26', 'Rpl27', 'Rpl27a', 'Rpl28', 'Rpl29', 'Rpl30', 'Rpl31', 'Rpl32', 'Rpl34',
    'Rpl35', 'Rpl35a', 'Rpl36', 'Rpl36a', 'Rpl37', 'Rpl37a', 'Rpl38', 'Rpl39', 'Rpl41', 'Rps2', 'Rps3',
    'Rps3a1', 'Rps4x', 'Rps5', 'Rps6', 'Rps7', 'Rps8', 'Rps9', 'Rps10', 'Rps11', 'Rps12', 'Rps13', 'Rps14',
    'Rps15', 'Rps15a', 'Rps16', 'Rps17', 'Rps18', 'Rps19', 'Rps20', 'Rps21', 'Rps23', 'Rps24', 'Rps25',
    'Rps26', 'Rps27', 'Rps27a', 'Rps28', 'Rps29']
NUCLEAR_LNCRNA_GENES = [
    'Xist', 'Malat1', 'Tsix', 'Nrg3os', 'Sorbs2os', 'Uckl1os', '1700024B05Rik', '1700024B18Rik',
    '1700085D07Rik', '1700111E14Rik', '4930415C11Rik', '4930419G24Rik', '4933413L06Rik', 'D830024N08Rik',
    'Gm12027', 'Gm13883', 'Gm13944', 'Gm13963', 'Gm14330', 'Gm15478', 'Gm15520', 'Gm15563', 'Gm16351',
    'Gm19710', 'Gm1992', 'Gm20275', 'Gm20642', 'Gm21954', 'Gm26699', 'Gm26798', 'Gm26804', 'Gm26854',
    'Gm26905', 'Gm28198', 'Gm28375', 'Gm28376', 'Gm32250', 'Gm35188', 'Gm36975', 'Gm37679', 'Gm42439',
    'Gm42477', 'Gm48678', 'Gm48742', 'Gm6994']
CYTOPLASMIC_GENES = ['Actb', 'Tmsb4x', 'Actg1', 'Calm1', 'Calm2', 'Calm3', 'Camk2n1', 'Eef1a1', 'Eef2', 'Eif4a2',
                     'Vamp2', 'Snap25', 'Syt1', 'Nrgn', 'Ywhab', 'Ywhaz', 'Ywhah', 'Ywhag']
NEURONAL_MARKER_GENES = ['Snap25', 'Syt1', 'Nrgn', 'Camk2n1', 'Vamp2', 'Rgs4', 'Calm1', 'Calm3']
NEURON_CT_PREFIXES = ['IT', 'ET', 'NP', 'CT', 'L6b', 'Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg']
GLIA_CT_PREFIXES = ['Astro', 'Oligo', 'OPC', 'Endo', 'Macrophage']

DISPLAY = {'ctmean': 'CT_mean', 'raw_nuclei': 'Raw_nuclei', 'linreg': 'Linear_reg',
           'linreg_shrunk': 'Linear_reg_shrunk', 'cvae': 'ImputeVAE', 'scvi': 'scVI', 'scanvi': 'scANVI',
           'scgen': 'scGen', 'scbutterfly': 'scButterfly'}


def df_to_markdown(df, fmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(["dir"] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else fmt.format(v))
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def base_name(dirname):
    return re.sub(r"_seed\d+$", "", dirname)


# ----------------------------------------------------------------------------
# Vectorised helpers
# ----------------------------------------------------------------------------

def ct_means(X, labels, cts):
    return np.array([X[labels == ct].mean(axis=0) for ct in cts], dtype=np.float64)   # [K, G]


def colwise_pearson(A, B):
    """Pearson r between columns of A and B ([K, G] each). NaN where a column is constant."""
    A = A - A.mean(axis=0); B = B - B.mean(axis=0)
    num = (A * B).sum(axis=0)
    den = np.sqrt((A ** 2).sum(axis=0) * (B ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[den <= 1e-12] = np.nan
    return r


def colwise_r2(pred, truth):
    ss_res = ((pred - truth) ** 2).sum(axis=0)
    ss_tot = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    r2[ss_tot <= 1e-12] = np.nan
    return r2


def summarize(v):
    v = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return {"median": np.nan, "mean": np.nan, "q25": np.nan, "q75": np.nan, "n_valid": 0, "n_total": int(v.size)}
    return {"median": float(np.median(v[ok])), "mean": float(np.mean(v[ok])),
            "q25": float(np.percentile(v[ok], 25)), "q75": float(np.percentile(v[ok], 75)),
            "n_valid": int(ok.sum()), "n_total": int(v.size)}


def safe_pearson(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(stats.pearsonr(a, b)[0])


def safe_spearman(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(stats.spearmanr(a, b)[0])


# ----------------------------------------------------------------------------
# A. cell-type-level
# ----------------------------------------------------------------------------

def ct_level_metrics(imp_M, cell_M, nuc_M, cts):
    """All inputs [K, G] means over the same cts."""
    out = {}
    gc = colwise_pearson(imp_M, cell_M)
    out["gene_corr"] = summarize(gc)
    out["gene_corr"]["pct_above_0.8"] = float(np.nanmean(gc > 0.8) * 100)
    out["r2"] = summarize(colwise_r2(imp_M, cell_M))
    out["gene_mse"] = summarize(((imp_M - cell_M) ** 2).mean(axis=0))
    prof_cell = np.array([safe_pearson(imp_M[k], cell_M[k]) for k in range(len(cts))])
    prof_nuc = np.array([safe_pearson(imp_M[k], nuc_M[k]) for k in range(len(cts))])
    out["profile_corr_to_cell"] = {"per_ct": dict(zip(cts, prof_cell.tolist())), "mean": float(np.nanmean(prof_cell))}
    out["shift"] = {"per_ct": dict(zip(cts, (prof_cell - prof_nuc).tolist())),
                    "mean": float(np.nanmean(prof_cell - prof_nuc)), "median": float(np.nanmedian(prof_cell - prof_nuc))}
    # delta
    emp = (cell_M - nuc_M); pred = (imp_M - nuc_M)
    e, p_ = emp.ravel(), pred.ravel()
    d = {"pearson": safe_pearson(e, p_), "spearman": safe_spearman(e, p_)}
    ss_tot = ((e - e.mean()) ** 2).sum()
    d["r2"] = float(1 - ((p_ - e) ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan
    d["sign_accuracy_all"] = float((np.sign(e) == np.sign(p_)).mean())
    big = np.abs(e) > 0.1
    d["sign_accuracy_abs_gt_0.1"] = float((np.sign(e[big]) == np.sign(p_[big])).mean()) if big.any() else np.nan
    d["n_abs_gt_0.1"] = int(big.sum())
    try:
        from sklearn.metrics import roc_auc_score
        nz = e != 0
        d["auroc_positive_delta"] = float(roc_auc_score((e[nz] > 0).astype(int), p_[nz]))
    except Exception:
        d["auroc_positive_delta"] = np.nan
    d["per_ct_pearson"] = {ct: safe_pearson(emp[k], pred[k]) for k, ct in enumerate(cts)}
    d["median_ct_pearson"] = float(np.nanmedian(list(d["per_ct_pearson"].values())))
    out["delta"] = d
    return out, gc


# ----------------------------------------------------------------------------
# B. within-CT distribution
# ----------------------------------------------------------------------------

def within_ct_distribution(imputed, ct_imp, X_cell, ct_cell, cts, hvg_idx, n_q=200):
    out = {"variance_ratio_per_ct": {}, "std_ratio_per_ct": {}, "wasserstein_per_ct_hvg": {},
           "qq_corr_per_ct_hvg": {}}
    w_all_genes = []   # per CT: [G] vector of W1
    qs = np.linspace(0.0025, 0.9975, n_q)
    weights = []
    for ct in cts:
        mi, mc = ct_imp == ct, ct_cell == ct
        if mi.sum() < 20 or mc.sum() < 20:
            continue
        A, B = imputed[mi], X_cell[mc]
        va, vb = A.var(axis=0), B.var(axis=0)
        out["variance_ratio_per_ct"][ct] = float(va.mean() / vb.mean()) if vb.mean() > 0 else np.nan
        ok = vb > 1e-10
        out["std_ratio_per_ct"][ct] = float(np.median(np.sqrt(va[ok] / vb[ok]))) if ok.any() else np.nan
        qa = np.quantile(A, qs, axis=0); qb = np.quantile(B, qs, axis=0)          # [n_q, G]
        w1 = np.abs(qa - qb).mean(axis=0)                                           # [G]
        w_all_genes.append(w1); weights.append(mi.sum())
        out["wasserstein_per_ct_hvg"][ct] = float(np.median(w1[hvg_idx]))
        qq = colwise_pearson(qa[:, hvg_idx], qb[:, hvg_idx])
        out["qq_corr_per_ct_hvg"][ct] = float(np.nanmedian(qq))
    vr = list(out["variance_ratio_per_ct"].values())
    out["variance_ratio"] = summarize(vr)
    out["std_ratio"] = summarize(list(out["std_ratio_per_ct"].values()))
    if w_all_genes:
        W = np.average(np.array(w_all_genes), axis=0, weights=np.array(weights, dtype=float))   # [G] nucleus-weighted
        out["wasserstein_all_genes"] = summarize(W)
        out["wasserstein_hvg"] = summarize(W[hvg_idx])
        out["qq_corr_hvg"] = summarize(list(out["qq_corr_per_ct_hvg"].values()))
    out["n_cts_used"] = len(vr)
    return out


def pca_space(X_train_cell, n_comp=50, seed=0):
    from sklearn.decomposition import PCA
    mu = X_train_cell.mean(axis=0); sd = X_train_cell.std(axis=0) + 1e-6
    pca = PCA(n_components=n_comp, random_state=seed, svd_solver="randomized")
    pca.fit((X_train_cell - mu) / sd)
    return lambda X: pca.transform((X - mu) / sd), pca


def energy_distance(A, B):
    dab = cdist(A, B).mean(); daa = cdist(A, A).mean(); dbb = cdist(B, B).mean()
    return float(2 * dab - daa - dbb)


def two_sample_auroc(A, B, seed=0):
    """Classifier two-sample test: 5-fold CV AUROC of a random forest telling real (A) from
    imputed (B) in PCA space.  0.5 = indistinguishable.  Neither a linear classifier nor a kNN
    detects a collapsed point mass sitting at the centre of the real cloud (linear: not separable;
    kNN: in 50-D the centroid is the nearest neighbour of almost every real cell, so all
    neighbourhoods look imputed).  Axis-aligned tree splits isolate it immediately."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    pr = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, pr))


def latent_space_metrics(P_imp, ct_imp, P_cell, ct_cell, cts, max_n=1000, seed=0):
    rng = np.random.RandomState(seed)
    out = {"energy_per_ct": {}, "energy_floor_per_ct": {}, "auroc_per_ct": {}, "auroc_floor_per_ct": {}}
    for ct in cts:
        ia, ic = np.where(ct_imp == ct)[0], np.where(ct_cell == ct)[0]
        if len(ia) < 20 or len(ic) < 40:
            continue
        n = min(max_n, len(ia), len(ic) // 2)
        a = P_imp[rng.choice(ia, n, replace=False)]
        cperm = rng.permutation(ic)
        c1, c2 = P_cell[cperm[:n]], P_cell[cperm[n:2 * n]]
        out["energy_per_ct"][ct] = energy_distance(a, c1)
        out["energy_floor_per_ct"][ct] = energy_distance(c2, c1)
        out["auroc_per_ct"][ct] = two_sample_auroc(c1, a, seed)
        out["auroc_floor_per_ct"][ct] = two_sample_auroc(c1, c2, seed)
    for k in ["energy", "energy_floor", "auroc", "auroc_floor"]:
        v = list(out[f"{k}_per_ct"].values())
        out[k] = {"median": float(np.median(v)) if v else np.nan, "mean": float(np.mean(v)) if v else np.nan}
    return out


# ----------------------------------------------------------------------------
# C. per-cell information retention
# ----------------------------------------------------------------------------

def rank_cols(X):
    return np.apply_along_axis(stats.rankdata, 0, X)


def within_ct_spearman(imputed, X_nuc, ct, cts, gene_idx):
    per_ct = {}; allv = []
    for c in cts:
        m = ct == c
        if m.sum() < 20:
            continue
        A, B = imputed[m][:, gene_idx], X_nuc[m][:, gene_idx]
        ok = (A.std(axis=0) > 1e-10) & (B.std(axis=0) > 1e-10)
        if ok.sum() < 10:
            continue
        r = colwise_pearson(rank_cols(A[:, ok]), rank_cols(B[:, ok]))
        per_ct[c] = float(np.nanmedian(r)); allv.extend(r[np.isfinite(r)].tolist())
    return {"per_ct": per_ct, **summarize(allv)}


def variance_rank_spearman(imputed, ct_imp, X_cell, ct_cell, cts):
    per_ct = {}
    for c in cts:
        mi, mc = ct_imp == c, ct_cell == c
        if mi.sum() < 20 or mc.sum() < 20:
            continue
        per_ct[c] = safe_spearman(imputed[mi].var(axis=0), X_cell[mc].var(axis=0))
    return {"per_ct": per_ct, **summarize(list(per_ct.values()))}


# ----------------------------------------------------------------------------
# D. downstream
# ----------------------------------------------------------------------------

def classification(P_train, y_train, P_test, y_test, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    clf.fit(P_train, y_train)
    pred = clf.predict(P_test)
    per_ct = {}
    for c in sorted(set(y_test)):
        m = y_test == c
        per_ct[c] = float((pred[m] == c).mean())
    return {"accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")), "per_ct_recall": per_ct}


def de_markers(X, labels, cts, top_k=100):
    """One-vs-rest markers per cell type.  Returns {"wilcoxon": {ct: set}, "logfc": {ct: set}}:
    top_k genes by tie-corrected rank-sum z (dispersion-sensitive) and top_k by mean difference
    in-group minus out-group on the log1p scale (dispersion-insensitive)."""
    N, G = X.shape
    R = rank_cols(X.astype(np.float64))
    tie = np.zeros(G)
    for g in range(G):
        _, cnt = np.unique(X[:, g], return_counts=True)
        tie[g] = (cnt ** 3 - cnt).sum()
    tot = X.sum(axis=0, dtype=np.float64)
    out = {"wilcoxon": {}, "logfc": {}}
    for ct in cts:
        m = labels == ct
        n1, n2 = m.sum(), N - m.sum()
        if n1 < 5 or n2 < 5:
            continue
        r1 = R[m].sum(axis=0)
        mu = n1 * (N + 1) / 2.0
        sigma = np.sqrt(n1 * n2 / 12.0 * ((N + 1) - tie / (N * (N - 1))))
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (r1 - mu) / sigma
        z[~np.isfinite(z)] = -np.inf
        out["wilcoxon"][ct] = set(np.argsort(-z)[:top_k].tolist())
        s_in = X[m].sum(axis=0, dtype=np.float64)
        fc = s_in / n1 - (tot - s_in) / n2
        out["logfc"][ct] = set(np.argsort(-fc)[:top_k].tolist())
    return out


def de_overlap(markers_ref, markers_test, top_k=100):
    per_ct = {ct: len(markers_ref[ct] & markers_test[ct]) / top_k for ct in markers_ref if ct in markers_test}
    return {"per_ct": per_ct, "mean": float(np.mean(list(per_ct.values()))) if per_ct else np.nan}


def coexpression(X_ref, X_test, idx):
    cr = np.corrcoef(X_ref[:, idx].T); ct_ = np.corrcoef(X_test[:, idx].T)
    iu = np.triu_indices(len(idx), 1)
    a, b = cr[iu], ct_[iu]
    ok = np.isfinite(a) & np.isfinite(b)
    return safe_spearman(a[ok], b[ok]) if ok.sum() > 10 else np.nan


# ----------------------------------------------------------------------------
# E. compartment biology
# ----------------------------------------------------------------------------

def compartment(imputed, X_nuc, ct_nuc, gene_names, cts):
    gl = list(gene_names)
    delta = imputed.mean(axis=0) - X_nuc.mean(axis=0)
    out = {}
    for name, gs, sgn in [("ribosomal", RIBOSOMAL_GENES, 1), ("cytoplasmic", CYTOPLASMIC_GENES, 1),
                          ("nuclear_lncrna", NUCLEAR_LNCRNA_GENES, -1)]:
        idx = [gl.index(g) for g in gs if g in gl]
        d = delta[idx]
        out[name] = {"n_found": len(idx), "mean_delta": float(d.mean()),
                     "frac_correct_sign": float(((d > 0) if sgn > 0 else (d < 0)).mean())}
    nidx = [gl.index(g) for g in NEURONAL_MARKER_GENES if g in gl]
    neu = [c for c in cts if any(p in c for p in NEURON_CT_PREFIXES)]
    gli = [c for c in cts if any(p in c for p in GLIA_CT_PREFIXES)]

    def _d(group):
        v = []
        for c in group:
            m = ct_nuc == c
            if m.sum() >= 5:
                v.append(imputed[m][:, nidx].mean() - X_nuc[m][:, nidx].mean())
        return float(np.mean(v)) if v else np.nan
    dn, dg = _d(neu), _d(gli)
    out["neuronal_markers"] = {"delta_neurons": dn, "delta_glia": dg,
                               "ratio": float(dn / dg) if (np.isfinite(dg) and dg != 0) else np.nan}
    return out


# ----------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------

def bootstrap(imputed, ct_imp, X_cell, ct_cell, X_nuc, cts, n_boot, seed=0):
    rng = np.random.RandomState(seed)
    idx_i = {c: np.where(ct_imp == c)[0] for c in cts}
    idx_c = {c: np.where(ct_cell == c)[0] for c in cts}
    rec = {"gene_corr_median": [], "r2_median": [], "delta_pearson": [], "delta_sign_acc": [],
           "variance_ratio_median": [], "shift_mean": []}
    for b in range(n_boot):
        imp_M, cell_M, nuc_M, vr = [], [], [], []
        for c in cts:
            si = rng.choice(idx_i[c], len(idx_i[c]), replace=True)
            sc = rng.choice(idx_c[c], len(idx_c[c]), replace=True)
            A, B = imputed[si], X_cell[sc]
            imp_M.append(A.mean(axis=0)); cell_M.append(B.mean(axis=0)); nuc_M.append(X_nuc[si].mean(axis=0))
            if len(si) >= 20 and len(sc) >= 20:
                vb = B.var(axis=0).mean()
                vr.append(A.var(axis=0).mean() / vb if vb > 0 else np.nan)
        imp_M, cell_M, nuc_M = map(np.array, (imp_M, cell_M, nuc_M))
        rec["gene_corr_median"].append(np.nanmedian(colwise_pearson(imp_M, cell_M)))
        rec["r2_median"].append(np.nanmedian(colwise_r2(imp_M, cell_M)))
        e, p_ = (cell_M - nuc_M).ravel(), (imp_M - nuc_M).ravel()
        rec["delta_pearson"].append(safe_pearson(e, p_))
        rec["delta_sign_acc"].append(float((np.sign(e) == np.sign(p_)).mean()))
        rec["variance_ratio_median"].append(np.nanmedian(vr) if vr else np.nan)
        pc = [safe_pearson(imp_M[k], cell_M[k]) for k in range(len(cts))]
        pn = [safe_pearson(imp_M[k], nuc_M[k]) for k in range(len(cts))]
        rec["shift_mean"].append(float(np.nanmean(np.array(pc) - np.array(pn))))
    out = {}
    for k, v in rec.items():
        v = np.array(v, dtype=float)
        out[k] = {"ci_low": float(np.nanpercentile(v, 2.5)), "ci_high": float(np.nanpercentile(v, 97.5)),
                  "sd": float(np.nanstd(v))}
    out["n_boot"] = n_boot
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def evaluate_method(name, imputed, ref, args):
    t0 = time.time()
    X_cell, ct_cell = ref["X_cell"], ref["ct_cell"]
    X_nuc, ct_nuc = ref["X_nuc"], ref["ct_nuc"]
    cts_all, cts_robust = ref["cts_all"], ref["cts_robust"]
    gene_names = ref["gene_names"]
    res = {"method": name, "n_eval_nuclei": int(imputed.shape[0])}

    imp_M = ct_means(imputed, ct_nuc, cts_all)
    res["ct_level"], gc = ct_level_metrics(imp_M, ref["cell_M"], ref["nuc_M"], cts_all)
    rob = [cts_all.index(c) for c in cts_robust]
    res["ct_level_robust"], _ = ct_level_metrics(imp_M[rob], ref["cell_M"][rob], ref["nuc_M"][rob], cts_robust)
    res["ct_level_robust"]["cts"] = cts_robust
    print(f"  [A] gene_corr {res['ct_level']['gene_corr']['median']:.3f} | r2 {res['ct_level']['r2']['median']:.3f} | "
          f"delta r {res['ct_level']['delta']['pearson']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    res["within_ct"] = within_ct_distribution(imputed, ct_nuc, X_cell, ct_cell, cts_all, ref["hvg_idx"])
    print(f"  [B] var ratio {res['within_ct']['variance_ratio']['median']:.3f} | W1(hvg) "
          f"{res['within_ct'].get('wasserstein_hvg', {}).get('median', np.nan):.3f} ({time.time()-t0:.0f}s)", flush=True)

    P_imp = ref["project"](imputed)
    res["latent"] = latent_space_metrics(P_imp, ct_nuc, ref["P_cell"], ct_cell, cts_all, seed=args.seed)
    print(f"  [B] energy {res['latent']['energy']['median']:.3f} (floor {res['latent']['energy_floor']['median']:.3f}) | "
          f"2-sample AUROC {res['latent']['auroc']['median']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    res["within_ct_spearman_vs_input"] = within_ct_spearman(imputed, X_nuc, ct_nuc, cts_all, ref["spearman_gene_idx"])
    res["variance_rank_spearman"] = variance_rank_spearman(imputed, ct_nuc, X_cell, ct_cell, cts_all)
    print(f"  [C] within-CT Spearman vs input {res['within_ct_spearman_vs_input']['median']:.3f} | var-rank rho "
          f"{res['variance_rank_spearman']['median']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    res["classification"] = classification(ref["P_train"], ref["ct_train"], P_imp, ct_nuc, seed=args.seed)
    if not args.skip_de:
        mk = de_markers(imputed, ct_nuc, cts_all)
        res["de_overlap_top100"] = de_overlap(ref["markers_cell"]["wilcoxon"], mk["wilcoxon"])
        res["de_overlap_top100_logfc"] = de_overlap(ref["markers_cell"]["logfc"], mk["logfc"])
    res["coexpression"] = {
        "pooled_top500": coexpression(X_cell, imputed, ref["hvg_idx"]),
        f"within_{ref['largest_ct']}_top500": coexpression(X_cell[ct_cell == ref["largest_ct"]],
                                                            imputed[ct_nuc == ref["largest_ct"]], ref["hvg_idx"]),
    }
    print(f"  [D] clf acc {res['classification']['accuracy']:.3f} | DE overlap "
          f"{res.get('de_overlap_top100', {}).get('mean', np.nan):.3f} (logFC {res.get('de_overlap_top100_logfc', {}).get('mean', np.nan):.3f}) | coexp {res['coexpression']['pooled_top500']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    res["compartment"] = compartment(imputed, X_nuc, ct_nuc, gene_names, cts_all)
    if args.n_boot > 0:
        res["bootstrap"] = bootstrap(imputed, ct_nuc, X_cell, ct_cell, X_nuc, cts_all, args.n_boot, seed=args.seed)
    res["eval_time_s"] = time.time() - t0
    print(f"  done in {res['eval_time_s']:.0f}s")
    return res, gc


def build_reference(args):
    print("Loading reference data from cache...")
    cd = args.cache_dir
    X_train, obs_train = load_partition("cell", "train", "log1p", cache_dir=cd)
    X_cell, obs_cell = load_partition("cell", args.eval_partition, "log1p", cache_dir=cd)
    X_nuc, obs_nuc = load_partition("nuc", args.eval_partition, "log1p", cache_dir=cd)
    gene_names = load_gene_names(cache_dir=cd)
    ct_train = obs_train["cell_type"].values.astype(str)
    ct_cell = obs_cell["cell_type"].values.astype(str)
    ct_nuc = obs_nuc["cell_type"].values.astype(str)
    cts_all = sorted(set(ct_cell) & set(ct_nuc))
    cts_robust = [c for c in cts_all if (ct_cell == c).sum() >= args.min_n_robust and (ct_nuc == c).sum() >= args.min_n_robust]
    print(f"  train cells {X_train.shape}; eval cells {X_cell.shape}; eval nuclei {X_nuc.shape}")
    print(f"  {len(cts_all)} common cell types; {len(cts_robust)} with >= {args.min_n_robust} in both modalities")
    counts = {c: {"n_cell_eval": int((ct_cell == c).sum()), "n_nuc_eval": int((ct_nuc == c).sum()),
                  "n_cell_train": int((ct_train == c).sum())} for c in cts_all}

    hvg_idx = np.argsort(X_train.var(axis=0))[-500:]
    rng = np.random.RandomState(args.seed)
    spearman_gene_idx = np.sort(rng.choice(X_train.shape[1], min(2000, X_train.shape[1]), replace=False))
    project, pca = pca_space(X_train, n_comp=50, seed=args.seed)
    P_train, P_cell = project(X_train), project(X_cell)
    largest_ct = max(cts_all, key=lambda c: (ct_nuc == c).sum())
    ref = {"X_cell": X_cell, "ct_cell": ct_cell, "X_nuc": X_nuc, "ct_nuc": ct_nuc, "obs_nuc": obs_nuc,
           "ct_train": ct_train, "gene_names": gene_names, "cts_all": cts_all, "cts_robust": cts_robust,
           "cell_M": ct_means(X_cell, ct_cell, cts_all), "nuc_M": ct_means(X_nuc, ct_nuc, cts_all),
           "hvg_idx": hvg_idx, "spearman_gene_idx": spearman_gene_idx, "project": project,
           "P_train": P_train, "P_cell": P_cell, "largest_ct": largest_ct, "counts": counts,
           "pca_explained": float(pca.explained_variance_ratio_.sum())}
    if not args.skip_de:
        t0 = time.time()
        ref["markers_cell"] = de_markers(X_cell, ct_cell, cts_all)
        print(f"  reference DE markers computed in {time.time()-t0:.0f}s")
    del X_train
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_partition", default="test", choices=["valid", "test"])
    ap.add_argument("--base_dir", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--methods", default="auto", help="comma list of method dirs, or 'auto'")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--min_n_robust", type=int, default=20)
    ap.add_argument("--skip_de", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=CACHE_DIR)
    args = ap.parse_args()
    base_dir = args.base_dir or f"run/benchmark/{args.eval_partition}"
    out_dir = args.output_dir or os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    if args.methods == "auto":
        dirs = sorted(d for d in os.listdir(base_dir) if os.path.exists(os.path.join(base_dir, d, "imputed_eval.npy")))
    else:
        dirs = [m.strip() for m in args.methods.split(",")]
    print(f"Evaluating {len(dirs)} method directories on the {args.eval_partition} partition: {dirs}")

    ref = build_reference(args)
    n_nuc = ref["X_nuc"].shape[0]
    all_results, per_gene = [], {}
    for d in dirs:
        mdir = os.path.join(base_dir, d)
        print(f"\n{'='*60}\n  {d}\n{'='*60}")
        imputed = np.load(os.path.join(mdir, "imputed_eval.npy")).astype(np.float32)
        obs = pd.read_csv(os.path.join(mdir, "nuc_eval_obs.csv"), dtype={"obs_id": str})
        assert imputed.shape == (n_nuc, len(ref["gene_names"])), (d, imputed.shape)
        assert (obs["obs_id"].values == ref["obs_nuc"]["obs_id"].values).all(), f"{d}: eval nuclei order differs from cache"
        gn = pd.read_csv(os.path.join(mdir, "gene_names.csv"), header=None)[0].astype(str).values
        assert np.array_equal(gn, ref["gene_names"]), f"{d}: gene order differs"
        meta = json.load(open(os.path.join(mdir, "arguments.json"))) if os.path.exists(os.path.join(mdir, "arguments.json")) else {}
        res, gc = evaluate_method(d, imputed, ref, args)
        res["display"] = DISPLAY.get(base_name(d), base_name(d))
        res["base_method"] = base_name(d)
        res["seed"] = meta.get("seed")
        res["n_params"] = meta.get("n_params")
        res["training_time_s"] = meta.get("training_time_s")
        res["checkpoint"] = meta.get("checkpoint")
        res["hyperparameters"] = {k: v for k, v in meta.items() if k not in ("library_by_ct",)}
        all_results.append(res)
        per_gene[d] = gc

    # ---- tables ----
    rows = []
    ceiling = next((r for r in all_results if r["base_method"] == "ctmean"), None)
    for r in all_results:
        A, Ar, B, L = r["ct_level"], r["ct_level_robust"], r["within_ct"], r["latent"]
        bs = r.get("bootstrap", {})

        def ci(k):
            return f"[{bs[k]['ci_low']:.3f}, {bs[k]['ci_high']:.3f}]" if k in bs else ""
        row = {
            "dir": r["method"], "Method": r["display"], "seed": r["seed"],
            "gene_corr_median": A["gene_corr"]["median"], "gene_corr_CI": ci("gene_corr_median"),
            "gene_corr_frac_of_ceiling": (A["gene_corr"]["median"] / ceiling["ct_level"]["gene_corr"]["median"]) if ceiling else np.nan,
            "gene_corr_median_robustCTs": Ar["gene_corr"]["median"],
            "r2_median": A["r2"]["median"], "r2_CI": ci("r2_median"),
            "gene_mse_median": A["gene_mse"]["median"],
            "profile_corr_mean": A["profile_corr_to_cell"]["mean"], "shift_mean": A["shift"]["mean"],
            "delta_pearson": A["delta"]["pearson"], "delta_CI": ci("delta_pearson"),
            "delta_spearman": A["delta"]["spearman"], "delta_r2": A["delta"]["r2"],
            "delta_sign_acc": A["delta"]["sign_accuracy_all"], "delta_sign_acc_abs_gt_0.1": A["delta"]["sign_accuracy_abs_gt_0.1"],
            "delta_auroc": A["delta"]["auroc_positive_delta"],
            "variance_ratio_median": B["variance_ratio"]["median"], "variance_ratio_CI": ci("variance_ratio_median"),
            "std_ratio_median": B["std_ratio"]["median"],
            "wasserstein_hvg_median": B.get("wasserstein_hvg", {}).get("median", np.nan),
            "wasserstein_all_median": B.get("wasserstein_all_genes", {}).get("median", np.nan),
            "qq_corr_hvg_median": B.get("qq_corr_hvg", {}).get("median", np.nan),
            "energy_dist_median": L["energy"]["median"], "energy_floor_median": L["energy_floor"]["median"],
            "two_sample_auroc_median": L["auroc"]["median"], "two_sample_auroc_floor": L["auroc_floor"]["median"],
            "within_ct_spearman_vs_input": r["within_ct_spearman_vs_input"]["median"],
            "variance_rank_spearman": r["variance_rank_spearman"]["median"],
            "clf_accuracy": r["classification"]["accuracy"], "clf_macro_f1": r["classification"]["macro_f1"],
            "de_overlap_top100": r.get("de_overlap_top100", {}).get("mean", np.nan),
            "de_overlap_top100_logfc": r.get("de_overlap_top100_logfc", {}).get("mean", np.nan),
            "coexp_pooled": r["coexpression"]["pooled_top500"],
            "coexp_within_largest_ct": r["coexpression"][f"within_{ref['largest_ct']}_top500"],
            "ribo_sign": r["compartment"]["ribosomal"]["frac_correct_sign"],
            "cyto_sign": r["compartment"]["cytoplasmic"]["frac_correct_sign"],
            "lncrna_sign": r["compartment"]["nuclear_lncrna"]["frac_correct_sign"],
            "neuron_glia_ratio": r["compartment"]["neuronal_markers"]["ratio"],
            "n_params": r["n_params"], "training_time_s": r["training_time_s"],
        }
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("dir")
    summary.to_csv(os.path.join(out_dir, "benchmark_summary.csv"))

    num_cols = [c for c in summary.columns if summary[c].dtype != object and c not in ("seed",)]
    agg = summary.groupby("Method")[num_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.to_csv(os.path.join(out_dir, "benchmark_summary_by_method.csv"))

    for key, sub, fname in [("ct_level", ("delta", "per_ct_pearson"), "per_ct_delta_pearson"),
                            ("ct_level", ("shift", "per_ct"), "per_ct_shift"),
                            ("within_ct", ("variance_ratio_per_ct",), "per_ct_variance_ratio"),
                            ("within_ct", ("wasserstein_per_ct_hvg",), "per_ct_wasserstein_hvg"),
                            ("latent", ("energy_per_ct",), "per_ct_energy"),
                            ("latent", ("auroc_per_ct",), "per_ct_two_sample_auroc"),
                            ("classification", ("per_ct_recall",), "per_ct_clf_recall"),
                            ("within_ct_spearman_vs_input", ("per_ct",), "per_ct_within_ct_spearman")]:
        tab = {}
        for r in all_results:
            v = r.get(key, {})
            for s in sub:
                v = v.get(s, {}) if isinstance(v, dict) else {}
            tab[r["method"]] = v
        pd.DataFrame(tab).to_csv(os.path.join(out_dir, f"{fname}.csv"))
    pd.DataFrame(ref["counts"]).T.to_csv(os.path.join(out_dir, "eval_sample_counts.csv"))
    np.savez_compressed(os.path.join(out_dir, "per_gene_corr.npz"), gene_names=ref["gene_names"], **per_gene)

    def _ser(o):
        if isinstance(o, dict):
            return {str(k): _ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_ser(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    with open(os.path.join(out_dir, "benchmark_full.json"), "w") as f:
        json.dump(_ser(all_results), f, indent=1)
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(_ser({
            "eval_partition": args.eval_partition, "manifest": MANIFEST_PATH, "cache_dir": CACHE_DIR,
            "n_eval_nuclei": int(n_nuc), "n_eval_cells": int(ref["X_cell"].shape[0]),
            "common_cts": ref["cts_all"], "robust_cts": ref["cts_robust"], "min_n_robust": args.min_n_robust,
            "largest_ct": ref["largest_ct"], "pca_explained_variance_50": ref["pca_explained"],
            "n_boot": args.n_boot, "seed": args.seed, "methods": dirs,
            "notes": ["baselines fitted on the train partition only",
                      "all references from the eval partition only",
                      "CT-mean row = sampling-noise ceiling for cell-type-level metrics",
                      "energy_floor / two_sample_auroc_floor = real eval cells vs real eval cells"],
        }), f, indent=2)

    # ---- markdown summary ----
    main_cols = ["Method", "seed", "gene_corr_median", "gene_corr_CI", "r2_median", "delta_pearson", "delta_sign_acc_abs_gt_0.1",
                 "variance_ratio_median", "wasserstein_hvg_median", "energy_dist_median", "two_sample_auroc_median",
                 "within_ct_spearman_vs_input", "clf_accuracy", "de_overlap_top100", "de_overlap_top100_logfc", "coexp_within_largest_ct",
                 "neuron_glia_ratio", "n_params"]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(f"# Split-aware benchmark — {args.eval_partition} partition ({time.strftime('%Y-%m-%d')})\n\n")
        f.write(f"{n_nuc} eval nuclei, {ref['X_cell'].shape[0]} eval cells, {len(ref['cts_all'])} common cell types "
                f"({len(ref['cts_robust'])} with >= {args.min_n_robust} per modality). Bootstrap CIs: {args.n_boot} resamples.\n\n")
        f.write("Noise floors: energy distance real-vs-real = "
                f"{all_results[0]['latent']['energy_floor']['median']:.3f}; two-sample AUROC real-vs-real = "
                f"{all_results[0]['latent']['auroc_floor']['median']:.3f}.\n\n")
        f.write(df_to_markdown(summary[main_cols]))
        f.write("\n\n## By method (mean ± sd over seeds)\n\n")
        agg_cols = [c for c in main_cols if c in num_cols and c not in ("n_params",)]
        g = summary.groupby("Method")[agg_cols]
        mean, sd, cnt = g.mean(), g.std(), g.count()
        lines = ["| Method | n_seeds | " + " | ".join(agg_cols) + " |", "|---|---|" + "---|" * len(agg_cols)]
        order = sorted(mean.index, key=lambda m: -mean.loc[m, "gene_corr_median"] if np.isfinite(mean.loc[m, "gene_corr_median"]) else 0)
        for m in order:
            cells = []
            for c in agg_cols:
                v, e, n = mean.loc[m, c], sd.loc[m, c], cnt.loc[m, c]
                if not np.isfinite(v):
                    cells.append("")
                elif n > 1 and np.isfinite(e):
                    cells.append(f"{v:.3f} ± {e:.3f}")
                else:
                    cells.append(f"{v:.3f}")
            lines.append(f"| {m} | {int(cnt.loc[m, agg_cols[0]])} | " + " | ".join(cells) + " |")
        f.write("\n".join(lines) + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print("\n" + summary[main_cols].to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
