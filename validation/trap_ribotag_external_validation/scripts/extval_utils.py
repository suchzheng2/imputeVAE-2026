"""Shared helpers for the external validation pipeline (Python fallback for edgeR/limma).

TMM normalization follows Robinson & Oshlack 2010 (edgeR calcNormFactors defaults:
logratioTrim=0.3, sumTrim=0.05, doWeighting=True, Acutoff=-1e10, reference = library whose
upper quartile is closest to the mean upper quartile).
"""
import numpy as np
import pandas as pd
from scipy import stats


def bh(p):
    p = np.asarray(p, dtype=float); n = len(p); o = np.argsort(p); r = np.empty(n); r[o] = np.arange(1, n + 1)
    q = np.minimum.accumulate((p * n / r)[o][::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def filter_by_expr(counts, group, min_count=10, min_total=15, large_n=10, min_prop=0.7):
    """Approximation of edgeR::filterByExpr. counts: genes x samples DataFrame; group: Series."""
    lib = counts.sum(axis=0)
    median_lib = np.median(lib)
    cpm_cut = min_count / median_lib * 1e6
    cpm = counts / lib * 1e6
    n_min = group.value_counts().min()
    if n_min > large_n:
        n_min = large_n + (n_min - large_n) * min_prop
    keep = ((cpm >= cpm_cut).sum(axis=1) >= n_min) & (counts.sum(axis=1) >= min_total)
    return keep


def tmm_factors(counts, logratio_trim=0.3, sum_trim=0.05):
    X = counts.values.astype(float); lib = X.sum(axis=0)
    uq = np.array([np.quantile(X[:, j][X[:, j] > 0] / lib[j], 0.75) if (X[:, j] > 0).any() else 0 for j in range(X.shape[1])])
    ref = int(np.argmin(np.abs(uq - uq.mean())))
    f = np.ones(X.shape[1])
    for j in range(X.shape[1]):
        if j == ref:
            continue
        o, r = X[:, j], X[:, ref]; no, nr = lib[j], lib[ref]
        ok = (o > 0) & (r > 0)
        po, pr = o[ok] / no, r[ok] / nr
        M = np.log2(po / pr); A = 0.5 * np.log2(po * pr)
        v = (no - o[ok]) / no / o[ok] + (nr - r[ok]) / nr / r[ok]
        n = len(M); loL = int(np.floor(n * logratio_trim)) + 1; hiL = n + 1 - loL
        loS = int(np.floor(n * sum_trim)) + 1; hiS = n + 1 - loS
        rM = stats.rankdata(M, method="ordinal"); rA = stats.rankdata(A, method="ordinal")
        keep = (rM >= loL) & (rM <= hiL) & (rA >= loS) & (rA <= hiS)
        f[j] = 2 ** (np.sum(M[keep] / v[keep]) / np.sum(1 / v[keep]))
    f = f / np.exp(np.mean(np.log(f)))
    return pd.Series(f, index=counts.columns, name="tmm")


def log_cpm(counts, factors=None, prior=2.0):
    """edgeR::cpm(log=TRUE, prior.count=2) with TMM effective library sizes."""
    lib = counts.sum(axis=0).astype(float)
    if factors is not None:
        lib = lib * factors
    prior_scaled = prior * lib / lib.mean()
    return np.log2((counts + prior_scaled) / (lib + 2 * prior_scaled) * 1e6)


def ols_contrast(logexpr, design, contrast):
    """Per-gene OLS: logexpr (genes x samples), design (samples x p), contrast (p,). Returns
    DataFrame with log2fc, se, t, p, fdr using residual df = n - p (no variance moderation)."""
    Y = logexpr.values; Xd = design.values.astype(float); n, p = Xd.shape
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    B = (XtX_inv @ Xd.T @ Y.T)              # p x genes
    resid = Y.T - Xd @ B
    df = n - p
    s2 = (resid ** 2).sum(axis=0) / df
    c = np.asarray(contrast, dtype=float)
    est = c @ B
    se = np.sqrt(s2 * (c @ XtX_inv @ c))
    t = np.where(se > 0, est / np.where(se > 0, se, 1), 0.0)
    pv = 2 * stats.t.sf(np.abs(t), df=df)
    return pd.DataFrame({"log2fc": est, "se": se, "t": t, "p": pv, "fdr": bh(pv), "df": df}, index=logexpr.index)


def spearman_ci(x, y, n_boot=1000, seed=42):
    x = np.asarray(x); y = np.asarray(y); rng = np.random.default_rng(seed)
    rho = stats.spearmanr(x, y)[0]; n = len(x)
    boots = np.array([stats.spearmanr(x[i], y[i])[0] for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    return rho, np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)
