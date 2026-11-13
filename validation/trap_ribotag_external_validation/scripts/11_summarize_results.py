#!/usr/bin/env python3
"""Print the headline numbers from the final tables in a compact form (used to write the report)."""
import json
import pandas as pd
TASK = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/validation/trap_ribotag_external_validation"; TAB = f"{TASK}/outputs/tables"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
s = pd.read_csv(f"{TAB}/concordance_summary_GSE210071_GSE121199.tsv", sep="\t")
cols = ["dataset", "condition", "contrast", "method", "model_col", "filter", "n_genes", "spearman", "spearman_ci_lo", "spearman_ci_hi", "pearson", "theilsen_slope", "slope_ci_lo", "slope_ci_hi", "auroc_external_label", "sign_agree_centered_external_reliable", "top10pct_pos_OR", "top10pct_pos_p", "bottom10pct_neg_OR", "bottom10pct_neg_p", "matched_null_mean", "matched_perm_p", "rho_pred_vs_nucexpr"]
print("\n### PRIMARY (filter none)"); p = s[(s["filter"] == "none")]
print(p[(p.method.isin(["ImputeVAE", "GroundTruth"])) & (p.model_col.isin(["ALL", "POOL_neuronal"]))][cols].round(3).to_string(index=False))
print("\n### OTHER METHODS (global, ALL)"); print(p[~p.method.isin(["ImputeVAE", "GroundTruth"])][["dataset", "condition", "contrast", "method", "n_genes", "spearman", "auroc_external_label", "matched_perm_p"]].round(3).to_string(index=False))
print("\n### MODEL COLUMN SENSITIVITY (ImputeVAE)"); print(p[(p.method == "ImputeVAE")][["dataset", "condition", "contrast", "model_col", "spearman", "auroc_external_label"]].round(3).pivot_table(index=["dataset", "condition", "contrast"], columns="model_col", values="spearman").round(3).to_string())
print("\n### FILTERS (ImputeVAE)"); f = s[(s.method == "ImputeVAE") & (s["filter"] != "none")]
print(f.pivot_table(index=["dataset", "condition", "contrast", "model_col"], columns="filter", values="spearman").round(3).to_string())
print("\n### NUC45 + replicate"); print(pd.read_csv(f"{TAB}/concordance_nuc45_GSE210071_GSE121199.tsv", sep="\t").round(3).to_string(index=False))
print("\n### CONTROL GENES"); g = pd.read_csv(f"{TAB}/concordance_genes_GSE210071_GSE121199.tsv", sep="\t"); g = g[g.status == "control"]
print(g.pivot_table(index="gene", columns=["dataset", "condition", "contrast"], values="external_log2fc").round(2).to_string())
print("\n### H2"); h2 = pd.read_csv(f"{TAB}/profile_improvement_H2.tsv", sep="\t")
print(h2[(h2["filter"] == "none")][["dataset", "cell_type", "batch", "model_split", "n_replicates", "n_genes", "spearman_nucleus", "spearman_imputed", "spearman_realcell", "delta_similarity_spearman", "gene_boot_ci_lo", "gene_boot_ci_hi", "replicate_boot_ci_lo", "replicate_boot_ci_hi", "gap_closed_frac", "pearson_nucleus", "pearson_imputed", "pearson_realcell"]].round(3).to_string(index=False))
print("\n### H2 filters/quintiles (META/greenberg only)"); print(h2[(h2["filter"] != "none") & (h2.batch.isin(["META", "greenberg", "batch1", "batch2"]))][["dataset", "cell_type", "batch", "filter", "n_genes", "spearman_nucleus", "spearman_imputed", "spearman_realcell", "delta_similarity_spearman"]].round(3).to_string(index=False))
print("\n### H3"); h3 = pd.read_csv(f"{TAB}/translational_support_H3.tsv", sep="\t")
print(h3[["dataset", "cell_type", "batch", "n_genes", "spearman_ribotag_vs_pred_delta", "spearman_ribotag_vs_gt_delta", "spearman_ribotag_vs_nucleus_expr", "partial_spearman_pred_given_nucexpr", "top5pct_frac_in_top20_ribotag", "top5pct_matched_null_mean", "top5pct_matched_p", "top10pct_frac_in_top20_ribotag", "top10pct_matched_null_mean", "top10pct_matched_p", "top20pct_frac_in_top20_ribotag", "top20pct_matched_p"]].round(3).to_string(index=False))
print("\n### SPECIFICITY"); print(pd.read_csv(f"{TAB}/celltype_specificity.tsv", sep="\t").round(3).to_string(index=False))
print("\n### SECONDARY"); sec = pd.read_csv(f"{TAB}/concordance_summary_secondary.tsv", sep="\t")
print(sec[["dataset", "contrast", "contrast_type", "method", "model_col", "n_genes", "spearman", "matched_null_mean", "matched_perm_p", "partial_spearman_given_nucexpr", "auroc_external_label"]].round(3).to_string(index=False))
