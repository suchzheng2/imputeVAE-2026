"""
Per-donor breakdown of the cell-type-level benchmark metrics (donor hold-out ablation, 2026-11-04).

For every method directory (standard outputs of predict_cvae.py etc.) and every donor among the eval
nuclei, cell-type means are formed from that donor's eval nuclei only and compared with the eval-cell
means of the same types (all cell donors; nuclei and cells come from different animals). Reports the
same A-level metrics as evaluate_all.py (gene corr, R2, delta r, profile corr, shift) per donor, so a
model trained without nuclei donor 5 (--holdout_donors nuclei:5) can be read off against donors 1-4.

Usage:
    python -m analysis.benchmark.eval_by_donor --eval_partition test \
        --methods abl_donor5_ctmed,abl_donor5_ktqmap,cvae_libsize_armA,cvae_libsize_armA_knntypeq \
        --output_dir run/benchmark/test/results_ablations/by_donor
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.benchmark.data_split import load_partition, load_gene_names, CACHE_DIR
from analysis.benchmark.evaluate_all import ct_means, ct_level_metrics


def md_table(piv, fmt="{:.3f}"):
    cols = [str(c) for c in piv.columns]
    out = ["| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in piv.iterrows():
        out.append(f"| {idx} | " + " | ".join(fmt.format(v) if np.isfinite(v) else "nan" for v in row.values) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_partition", default="test", choices=["valid", "test"])
    ap.add_argument("--base_dir", default=None)
    ap.add_argument("--methods", required=True, help="comma list of method dirs under base_dir")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_n", type=int, default=20, help="min eval nuclei of a type within a donor, and min eval cells")
    ap.add_argument("--cache_dir", default=CACHE_DIR)
    args = ap.parse_args()
    base_dir = args.base_dir or f"run/benchmark/{args.eval_partition}"
    os.makedirs(args.output_dir, exist_ok=True)

    X_cell, obs_cell = load_partition("cell", args.eval_partition, "log1p", cache_dir=args.cache_dir)
    X_nuc, obs_nuc = load_partition("nuc", args.eval_partition, "log1p", cache_dir=args.cache_dir)
    gene_names = load_gene_names(cache_dir=args.cache_dir)
    ct_cell = obs_cell["cell_type"].values.astype(str)
    ct_nuc = obs_nuc["cell_type"].values.astype(str)
    donor_nuc = obs_nuc["donor"].values.astype(str)
    donors = sorted(set(donor_nuc), key=lambda d: (len(d), d))
    print(f"eval cells {X_cell.shape}, eval nuclei {X_nuc.shape}, nuclei donors {donors}")

    # one fixed type set per donor: types with >= min_n nuclei from that donor and >= min_n eval cells
    cts_by_donor = {}
    for d in donors + ["all"]:
        m = np.ones(len(ct_nuc), bool) if d == "all" else donor_nuc == d
        cts_by_donor[d] = sorted(c for c in set(ct_nuc[m])
                                 if (ct_nuc[m] == c).sum() >= args.min_n and (ct_cell == c).sum() >= args.min_n)
    # the common type set lets donors be compared on identical types
    cts_common = sorted(set.intersection(*[set(cts_by_donor[d]) for d in donors]))
    print(f"types per donor: { {d: len(v) for d, v in cts_by_donor.items()} }; common across donors: {len(cts_common)}")

    rows, full = [], {}
    for mname in [m.strip() for m in args.methods.split(",") if m.strip()]:
        mdir = os.path.join(base_dir, mname)
        imputed = np.load(os.path.join(mdir, "imputed_eval.npy")).astype(np.float32)
        obs = pd.read_csv(os.path.join(mdir, "nuc_eval_obs.csv"), dtype={"obs_id": str})
        assert (obs["obs_id"].values == obs_nuc["obs_id"].values).all(), f"{mname}: eval nuclei order differs from cache"
        meta = json.load(open(os.path.join(mdir, "arguments.json"))) if os.path.exists(os.path.join(mdir, "arguments.json")) else {}
        full[mname] = {"meta": {k: meta.get(k) for k in ("run_dir", "target_libsize", "uses_cell_type_label")}, "donors": {}}
        for d in donors + ["all"]:
            m = np.ones(len(ct_nuc), bool) if d == "all" else donor_nuc == d
            for label, cts in (("own_types", cts_by_donor[d]), ("common_types", cts_common)):
                if not cts:
                    continue
                imp_M = ct_means(imputed[m], ct_nuc[m], cts)
                cell_M = ct_means(X_cell, ct_cell, cts)
                nuc_M = ct_means(X_nuc[m], ct_nuc[m], cts)
                res, _ = ct_level_metrics(imp_M, cell_M, nuc_M, cts)
                full[mname]["donors"][f"{d}/{label}"] = res
                rows.append({"method": mname, "donor": d, "type_set": label, "n_nuclei": int(m.sum()), "n_types": len(cts),
                             "gene_corr_median": res["gene_corr"]["median"], "r2_median": res["r2"]["median"],
                             "delta_pearson": res["delta"]["pearson"], "delta_median_ct_pearson": res["delta"]["median_ct_pearson"],
                             "profile_corr_mean": res["profile_corr_to_cell"]["mean"], "shift_mean": res["shift"]["mean"]})
        r = [x for x in rows if x["method"] == mname and x["type_set"] == "common_types"]
        print(f"{mname}: " + " | ".join(f"donor {x['donor']} gc {x['gene_corr_median']:.3f} r2 {x['r2_median']:.3f} dr {x['delta_pearson']:.3f}" for x in r))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "by_donor.csv"), index=False)
    with open(os.path.join(args.output_dir, "by_donor_full.json"), "w") as f:
        json.dump({"cts_by_donor": cts_by_donor, "cts_common": cts_common, "min_n": args.min_n, "results": full}, f, indent=2, default=str)
    com = df[df.type_set == "common_types"]
    with open(os.path.join(args.output_dir, "by_donor.md"), "w") as f:
        f.write(f"# Per-donor breakdown, {args.eval_partition} partition\n\nCT-level metrics on the {len(cts_common)} types "
                f"with >= {args.min_n} eval nuclei in every donor ({', '.join(cts_common)}); the 'all' column pools the donors "
                f"on the same types. Numbers are lower than the main table because the rare (mostly non-neuronal) types that "
                f"carry most between-type variance are absent; compare across donors / methods, not with results/.\n\n")
        for title, val in (("Gene corr (median over genes)", "gene_corr_median"), ("R2 (median over genes)", "r2_median"),
                           ("delta Pearson", "delta_pearson"), ("profile corr to cell (mean over types)", "profile_corr_mean")):
            f.write(f"## {title}\n\n" + md_table(com.pivot(index="method", columns="donor", values=val)) + "\n\n")
        own = df[(df.type_set == "own_types") & (df.donor == "all")].set_index("method")
        f.write(f"## All donors pooled, every type with >= {args.min_n} eval nuclei and cells ({len(cts_by_donor['all'])} types)\n\n"
                + md_table(own[["gene_corr_median", "r2_median", "delta_pearson", "profile_corr_mean"]]) + "\n")
    print(f"wrote {args.output_dir}/by_donor.{{csv,md}}")


if __name__ == "__main__":
    main()
