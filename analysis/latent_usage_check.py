#!/usr/bin/env python3
"""Latent-usage check on the frozen full-data cVAE (no retraining).

Imputation decodes from mu only, so within-cell-type variation in the output can only come from within-cell-type
variation of mu. Questions:
  1. How many of the 32 latent dimensions are active (overall, and WITHIN cell type)?
  2. How much of the within-cell-type latent variation is library size?
  3. Which dimensions drive the within-cell-type variance of the imputed output (ablation: set one dimension to its
     cell-type mean and decode; also keep ONE dimension and freeze the rest)?
  4. Is within-cell-type variation of mu large relative to the posterior noise exp(logvar) the decoder saw in training?
Outputs -> notes/experiments/variance_zero_nonzero_fullmodel/latent_usage/
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imputeVAE.models_cond import VAE_JSD_Cond
from variance_zero_nonzero_fullmodel import load_counts, PROJECT

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run_dir", default=f"{PROJECT}/run/run_2026-07-18-19-40-51_570984")
ap.add_argument("--out", default=f"{PROJECT}/notes/experiments/variance_zero_nonzero_fullmodel/latent_usage")
ap.add_argument("--imputed", default=f"{PROJECT}/run/cvae_full_eval/imputed_cell_from_nucleus.npy",
                help="exported imputed matrix for the decode sanity check (analysis/export_cvae_imputed.py, same "
                     "--target_libsize as below for covariate models); 'none' skips the check (mid-training checkpoints)")
ap.add_argument("--target_libsize", default="ct_median_cell", choices=["ct_median_cell", "source"],
                help="libsize_cond models (2026-10-16): library the nuclei->cell decode uses in the ablations")
cli = ap.parse_args()
RUN = cli.run_dir
OUT = cli.out; os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = json.load(open(f"{RUN}/arguments.json"))
ni = cfg["num_internal"]; ni = [int(x) for x in ni.split(",")] if isinstance(ni, str) else ni
model = VAE_JSD_Cond(input_size=cfg["num_feature"], num_internal=ni, num_latent=cfg["num_latent"], embed_dim=cfg.get("embed_dim", 8),
                     scalar_cond=cfg.get("scalar_cond", False), cond_encoder=cfg.get("cond_encoder", False),
                     libsize_cond=cfg.get("libsize_cond", False))
model.load_state_dict(torch.load(f"{RUN}/models/model.pt", map_location="cpu")); model.eval().to(dev)
K = cfg["num_latent"]; print("run:", RUN, "latent dims:", K, "libsize_cond:", model.libsize_cond, "device:", dev, flush=True)


@torch.no_grad()
def encode(X, modality_id, bs=2048):
    mus, lvs = [], []
    for s in range(0, len(X), bs):
        x = torch.from_numpy(X[s:s + bs]).to(dev); emb = model._get_modality_emb(x.size(0), modality_id, dev)
        sl = model._standardize_libsize(model.log_library_size(x)) if model.libsize_cond else None
        h = model.encoder(model._encoder_input(x, emb, sl))
        mus.append(model.mu(h).cpu().numpy()); lvs.append(torch.clamp(model.log_var(h), -20, 20).cpu().numpy())
    return np.concatenate(mus), np.concatenate(lvs)


@torch.no_grad()
def decode(Z, modality_id, S=None, bs=4096):
    """S: per-row STANDARDISED target library (required for libsize_cond models, aligned with Z)."""
    if model.libsize_cond and S is None:
        raise ValueError("libsize_cond model: decode needs the standardised target library S")
    out = []
    for s in range(0, len(Z), bs):
        z = torch.from_numpy(Z[s:s + bs].astype(np.float32)).to(dev); emb = model._get_modality_emb(z.size(0), modality_id, dev)
        sl = torch.from_numpy(S[s:s + bs].astype(np.float32)).to(dev).view(-1, 1) if S is not None else None
        out.append(model.decoder(model._decoder_input(z, emb, sl)).cpu().numpy())
    return np.concatenate(out)


gene_list = pd.read_csv(f"{PROJECT}/data/training_gene_list_v3.csv", header=None)[0].tolist()
common_genes = pd.read_csv(f"{PROJECT}/data/common_genes.csv")["common_genes"].tolist()
cts = pd.read_csv(f"{PROJECT}/data/common_cell_types.csv")["common_cell_types"].tolist()
Nc, nct, genes = load_counts("10X_nuclei_v2_v3_AIBS", "n3A", gene_list, common_genes, cts)
Cc, cct, _ = load_counts("10X_cells_v2_v3_AIBS", "c3A", gene_list, common_genes, cts)
Ln, Lc = np.log1p(Nc), np.log1p(Cc); tot_n = Nc.sum(1); lib_n, lib_c = np.log(tot_n), np.log(Cc.sum(1)); del Nc, Cc
# decode target for the covariate model: standardised log1p library per nucleus (S_n), None for the paper model
S_n = None
if model.libsize_cond:
    if cli.target_libsize == "ct_median_cell":
        med = json.load(open(f"{RUN}/libsize_stats.json"))["ct_median_cell"]
        tgt = np.array([med[c] for c in nct], dtype=np.float64)
    else:
        tgt = np.log1p(tot_n)
    S_n = ((tgt - float(model.libsize_mean)) / float(model.libsize_std)).astype(np.float32)
    print(f"decode target = {cli.target_libsize}; standardised S_n mean {S_n.mean():.3f} sd {S_n.std():.3f}", flush=True)
mu_n, lv_n = encode(Ln, 1); mu_c, lv_c = encode(Lc, 0)          # 1 = nuclei, 0 = cell (as in export_cvae_imputed.py)
np.save(f"{OUT}/mu_nuclei.npy", mu_n.astype(np.float32)); np.save(f"{OUT}/mu_cells.npy", mu_c.astype(np.float32))
# sanity: decoding mu with the cell flag must reproduce the saved imputed matrix
chk = decode(mu_n[:2000], 0, None if S_n is None else S_n[:2000])
if cli.imputed.lower() == "none":
    decode_check = float("nan"); print("decode sanity check skipped (--imputed none)", flush=True)
else:
    I = np.load(cli.imputed, mmap_mode="r"); decode_check = float(np.abs(chk - I[:2000]).max())
    print("max |decode(mu) - saved imputed| on 2000 nuclei:", decode_check, flush=True)


def within(M, labels):
    Z = M.copy()
    for ct in np.unique(labels): Z[labels == ct] -= Z[labels == ct].mean(0)
    return Z


def pr(Z):
    ev = np.linalg.eigvalsh(np.cov(Z.T)); ev = np.clip(ev, 0, None); return float(ev.sum() ** 2 / (ev ** 2).sum())


def r2_each(x, Z):
    x = x - x.mean(); return (Z * x[:, None]).mean(0) ** 2 / (Z.var(0) * x.var() + 1e-20)


# ---------------- 1 + 2 + 4: per-dimension table (nuclei and cells)
rows = []
for nm, M, LV, lab, lib in (("nuclei", mu_n, lv_n, nct, lib_n), ("cells", mu_c, lv_c, cct, lib_c)):
    W = within(M, lab); libw = within(lib[:, None], lab)[:, 0]
    kl = 0.5 * (M ** 2 + np.exp(LV) - LV - 1).mean(0)
    for k in range(K):
        rows.append(dict(source=nm, dim=k, var_total=M[:, k].var(), var_within_ct=W[:, k].var(), var_between_ct=M[:, k].var() - W[:, k].var(),
                         mean_posterior_var=float(np.exp(LV[:, k]).mean()), snr_within=W[:, k].var() / float(np.exp(LV[:, k]).mean()), kl_to_std_normal=kl[k],
                         r2_libsize_within_ct=float(r2_each(libw, W[:, [k]])[0]), r_libsize_within_ct=float(np.corrcoef(libw, W[:, k])[0, 1])))
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/per_dimension.csv", index=False)

# ---------------- per cell type: latent participation ratio, library-size share
ct_rows = []
for ct in cts:
    for nm, M, lab, lib in (("nuclei", mu_n, nct, lib_n), ("cells", mu_c, cct, lib_c)):
        m = lab == ct
        if m.sum() < 50: continue
        Z = M[m] - M[m].mean(0); x = lib[m] - lib[m].mean(); v = Z.var(0)
        r2 = r2_each(x, Z); ct_rows.append(dict(cell_type=ct, source=nm, n=int(m.sum()), latent_participation_ratio=pr(Z), n_dims_var_gt_0p01=int((v > 0.01).sum()),
                                                  n_dims_for_90pct=int(np.searchsorted(np.cumsum(np.sort(v)[::-1]) / v.sum(), 0.9) + 1), top_dim=int(v.argmax()), top_dim_share=float(v.max() / v.sum()),
                                                  libsize_share_of_latent_var=float((r2 * v).sum() / v.sum()), top_libsize_dim=int(r2.argmax()), top_libsize_r2=float(r2.max())))
CT = pd.DataFrame(ct_rows); CT.to_csv(f"{OUT}/per_cell_type.csv", index=False)

# ---------------- 3: ablation of each dimension on the imputed OUTPUT (nuclei -> cell decode)
base = decode(mu_n, 0, S_n)
def within_var_sum(Y):
    tot = 0.0; per = {}
    for ct in np.unique(nct):
        m = nct == ct
        if m.sum() < 50: continue
        v = Y[m].var(0).sum() * m.sum(); per[ct] = Y[m].var(0).sum(); tot += v
    return tot / sum((nct == ct).sum() for ct in per), per
base_tot, base_per = within_var_sum(base)
ct_mean = {ct: mu_n[nct == ct].mean(0) for ct in np.unique(nct)}
CM = np.stack([ct_mean[c] for c in nct])
abl = []
for k in range(K):
    Z = mu_n.copy(); Z[:, k] = CM[:, k]; t_drop, per_drop = within_var_sum(decode(Z, 0, S_n))       # remove dim k
    Z = CM.copy(); Z[:, k] = mu_n[:, k]; t_keep, per_keep = within_var_sum(decode(Z, 0, S_n))       # keep ONLY dim k
    abl.append(dict(dim=k, share_lost_when_frozen=1 - t_drop / base_tot, share_retained_when_only_dim=t_keep / base_tot,
                    **{f"keep_only|{ct}": per_keep[ct] / base_per[ct] for ct in base_per}))
    print(f"dim {k:2d}: lost {abl[-1]['share_lost_when_frozen']:.3f}  only-this-dim {abl[-1]['share_retained_when_only_dim']:.3f}", flush=True)
A = pd.DataFrame(abl); A.to_csv(f"{OUT}/ablation_per_dimension.csv", index=False)
# cumulative: keep the top-m dims by single-dim share
order = A.sort_values("share_retained_when_only_dim", ascending=False).dim.tolist(); cum = []
for m_ in (1, 2, 3, 5, 8, 12, 16, 24, 32):
    Z = CM.copy(); Z[:, order[:m_]] = mu_n[:, order[:m_]]; t, _ = within_var_sum(decode(Z, 0, S_n)); cum.append(dict(n_dims_kept=m_, share_of_output_within_ct_variance=t / base_tot, dims=str(order[:m_])))
pd.DataFrame(cum).to_csv(f"{OUT}/ablation_cumulative.csv", index=False)

# ---------------- library size: how well do the latents predict it, and output variance after removing the lib direction
Wn = within(mu_n, nct); lw = within(lib_n[:, None], nct)[:, 0]
beta, *_ = np.linalg.lstsq(Wn, lw, rcond=None); r2_all = 1 - ((lw - Wn @ beta) ** 2).sum() / (lw ** 2).sum()
b = (Wn * lw[:, None]).mean(0) / lw.var()                         # latent shift per unit log-library within CT
Z = mu_n - np.outer(lw, b); t_nolib, _ = within_var_sum(decode(Z, 0, S_n))
summary = dict(run_dir=RUN, libsize_cond=bool(model.libsize_cond), decode_target=(cli.target_libsize if model.libsize_cond else "n/a"),
               latent_dims=K, decode_check_max_abs_diff=decode_check, r2_loglib_from_all_latents_within_ct=float(r2_all),
               share_output_within_ct_var_removed_by_projecting_out_libsize=float(1 - t_nolib / base_tot), output_within_ct_var_sum=float(base_tot),
               nuclei_dims_total_var_gt_0p01=int((D[D.source == "nuclei"].var_total > 0.01).sum()), nuclei_dims_within_var_gt_0p01=int((D[D.source == "nuclei"].var_within_ct > 0.01).sum()),
               nuclei_latent_pr_pooled_within=pr(Wn), cells_latent_pr_pooled_within=pr(within(mu_c, cct)), nuclei_latent_pr_total=pr(mu_n - mu_n.mean(0)))
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
pd.set_option("display.width", 220); print(json.dumps(summary, indent=2)); print(D[D.source == "nuclei"].sort_values("var_within_ct", ascending=False).round(4).head(12).to_string(index=False))
print(CT.round(3).to_string(index=False)); print(pd.DataFrame(cum).round(3).to_string(index=False))
