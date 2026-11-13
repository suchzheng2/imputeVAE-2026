"""
Is the latent free of sequencing-library (batch) structure?  (library covariate check, 2026-11-06)

For each run: encode the eval-partition nuclei (modality 1) and cells (modality 0) to posterior means
(library_cond models get each observation's own library on the encoder side), then within every
cell type and modality measure how much the latent still sorts by library:

  knn_library_purity   mean fraction of an observation's K nearest neighbours (same modality, same
                       type) that come from its own library; chance = sum_l p_l^2 (library proportions)
  excess_purity        purity - chance;  norm_excess = (purity - chance) / (1 - chance)  (0 = mixed,
                       1 = perfectly separated by library)
  silhouette_library   sklearn silhouette by library label (sampled to <= 2000 per group)
  knn_modality_purity  same statistic for nuclei-vs-cells of the type (cross-modality alignment)

Usage:
    python -m analysis.benchmark.latent_library_mixing \
        --run_dirs run/run_2026-10-16_libsize_armA_jsd0.1_seed1,run/run_2026-11-06_libcond_seed1 \
        --output_dir run/benchmark/test/results_libcond/latent_mixing
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.benchmark.data_split import load_benchmark_data
from analysis.benchmark.predict_cvae import load_model, load_library_vocab, library_ids_from_obs, encode_all


def knn_purity(Z, labels, k):
    """Mean own-label fraction among the K nearest neighbours (self excluded) and the chance level."""
    from sklearn.neighbors import NearestNeighbors
    labels = np.asarray(labels).astype(str)
    k = min(k, len(labels) - 1)
    idx = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    purity = float((labels[idx] == labels[:, None]).mean())
    p = pd.Series(labels).value_counts(normalize=True).to_numpy()
    chance = float((p ** 2).sum())
    return purity, chance


def silhouette(Z, labels, max_n=2000, seed=0):
    from sklearn.metrics import silhouette_score
    labels = np.asarray(labels).astype(str)
    if len(np.unique(labels)) < 2:
        return np.nan
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(labels), min(max_n, len(labels)), replace=False)
    return float(silhouette_score(Z[sel], labels[sel]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dirs", required=True, help="comma list of run folders")
    p.add_argument("--eval_partition", default="test", choices=["valid", "test"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--min_obs", type=int, default=50)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_benchmark_data(args.eval_partition, return_raw_counts=False, include_eval_cells=True)
    sets = {"nuclei": (data["nuc_eval"], 1), "cell": (data["cell_eval"], 0)}
    rows = []
    for run_dir in args.run_dirs.split(","):
        run_dir = run_dir.strip().rstrip("/")
        name = os.path.basename(run_dir)
        model, config, _ = load_model(run_dir, device)
        vocab = load_library_vocab(run_dir) if model.library_cond else None
        Z = {}
        for modality, (adata, mid) in sets.items():
            lib = library_ids_from_obs(vocab, adata.obs, modality) if vocab else None
            Z[modality] = encode_all(model, adata.X.astype(np.float32), mid, device, library_id=lib)
            print(f"{name}: encoded {Z[modality].shape[0]} {modality} (library_cond={model.library_cond})")
        for modality, (adata, _) in sets.items():
            obs = adata.obs
            for ct in sorted(obs["cell_type"].unique()):
                m = (obs["cell_type"] == ct).to_numpy()
                lab = obs.loc[m, "donor"].astype(str).to_numpy()
                if m.sum() < args.min_obs or len(np.unique(lab)) < 2:
                    continue
                pur, ch = knn_purity(Z[modality][m], lab, args.k)
                rows.append({"run": name, "modality": modality, "cell_type": ct, "n": int(m.sum()),
                             "n_libraries": int(len(np.unique(lab))), "knn_library_purity": pur, "chance": ch,
                             "excess_purity": pur - ch, "norm_excess": (pur - ch) / (1 - ch) if ch < 1 else np.nan,
                             "silhouette_library": silhouette(Z[modality][m], lab)})
        # cross-modality mixing per type
        for ct in sorted(set(sets["nuclei"][0].obs["cell_type"]) & set(sets["cell"][0].obs["cell_type"])):
            mn = (sets["nuclei"][0].obs["cell_type"] == ct).to_numpy()
            mc = (sets["cell"][0].obs["cell_type"] == ct).to_numpy()
            if mn.sum() < args.min_obs or mc.sum() < args.min_obs:
                continue
            Zc = np.concatenate([Z["nuclei"][mn], Z["cell"][mc]])
            lab = np.array(["nuclei"] * int(mn.sum()) + ["cell"] * int(mc.sum()))
            pur, ch = knn_purity(Zc, lab, args.k)
            rows.append({"run": name, "modality": "both", "cell_type": ct, "n": int(len(lab)), "n_libraries": 2,
                         "knn_library_purity": pur, "chance": ch, "excess_purity": pur - ch,
                         "norm_excess": (pur - ch) / (1 - ch), "silhouette_library": silhouette(Zc, lab)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "latent_library_mixing_per_type.csv"), index=False)
    # obs-weighted pooled summary
    def pooled(g):
        w = g["n"] / g["n"].sum()
        return pd.Series({"n_types": len(g), "knn_library_purity": (g["knn_library_purity"] * w).sum(),
                          "chance": (g["chance"] * w).sum(), "norm_excess": (g["norm_excess"] * w).sum(),
                          "silhouette_library": (g["silhouette_library"] * w).sum()})
    summ = df.groupby(["run", "modality"]).apply(pooled).reset_index()
    summ.to_csv(os.path.join(args.output_dir, "latent_library_mixing_summary.csv"), index=False)
    with open(os.path.join(args.output_dir, "latent_library_mixing.md"), "w") as f:
        f.write(f"# Latent library mixing ({args.eval_partition} partition, K={args.k})\n\n")
        f.write("norm_excess = (kNN own-library purity - chance) / (1 - chance): 0 = libraries fully mixed within a "
                "cell type, 1 = fully separated. Row `both` = nuclei-vs-cell purity (cross-modality alignment).\n\n")
        f.write("| run | modality | types | purity | chance | norm_excess | silhouette |\n|---|---|---|---|---|---|---|\n")
        for _, r in summ.iterrows():
            f.write(f"| {r['run']} | {r['modality']} | {int(r['n_types'])} | {r['knn_library_purity']:.3f} | "
                    f"{r['chance']:.3f} | {r['norm_excess']:.3f} | {r['silhouette_library']:.3f} |\n")
    print(summ.to_string(index=False))
    print(f"-> {args.output_dir}")


if __name__ == "__main__":
    main()
