"""Gene Ontology enrichment analysis on delta analysis gene lists.

Runs g:Profiler on top increased/decreased genes from the delta analysis
to identify enriched GO terms, KEGG pathways, and cellular compartments.

Usage:
    python analysis/go_analysis.py [--n_genes 100] [--output_dir <dir>]
"""

import argparse
import os
import pandas as pd
from gprofiler import GProfiler

DELTA_DIR = "notes/experiments/delta_analysis"


def run_go_enrichment(gene_list, organism="mmusculus", background=None, sources=None):
    """Run g:Profiler enrichment on a gene list."""
    if sources is None:
        sources = ["GO:BP", "GO:CC", "GO:MF", "KEGG", "REAC"]

    gp = GProfiler(return_dataframe=True)
    result = gp.profile(
        organism=organism,
        query=gene_list,
        background=background,
        sources=sources,
        significance_threshold_method="fdr",
    )
    return result


def format_results(df, max_rows=30):
    """Format g:Profiler results for display."""
    if df.empty:
        return "No significant enrichments found."

    cols = ["source", "native", "name", "p_value", "intersection_size", "term_size"]
    available = [c for c in cols if c in df.columns]
    subset = df[available].head(max_rows)
    return subset.to_string(index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_genes", type=int, default=100)
    parser.add_argument("--output_dir", default=os.path.join(DELTA_DIR, "go_enrichment"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    inc_file = os.path.join(DELTA_DIR, "top100_increased_genes.txt")
    dec_file = os.path.join(DELTA_DIR, "top100_decreased_genes.txt")
    bg_file = os.path.join(DELTA_DIR, "all_genes_background.txt")

    increased = pd.read_csv(inc_file, header=None)[0].tolist()[:args.n_genes]
    decreased = pd.read_csv(dec_file, header=None)[0].tolist()[:args.n_genes]
    background = pd.read_csv(bg_file, header=None)[0].tolist()

    print(f"Increased genes: {len(increased)}, Decreased genes: {len(decreased)}")
    print(f"Background: {len(background)} genes")

    # Filter out Gm* predicted genes from decreased list for cleaner GO
    decreased_named = [g for g in decreased if not g.startswith("Gm")]
    print(f"Decreased (excluding Gm*): {len(decreased_named)}")

    # Run enrichment
    print("\n=== INCREASED GENES (top {}) ===".format(args.n_genes))
    inc_results = run_go_enrichment(increased, background=background)
    if not inc_results.empty:
        inc_results.to_csv(os.path.join(args.output_dir, "increased_go_results.csv"), index=False)
        print(format_results(inc_results))
    else:
        print("No significant enrichments.")

    print("\n=== DECREASED GENES (top {}) ===".format(args.n_genes))
    dec_results = run_go_enrichment(decreased, background=background)
    if not dec_results.empty:
        dec_results.to_csv(os.path.join(args.output_dir, "decreased_go_results.csv"), index=False)
        print(format_results(dec_results))
    else:
        print("No significant enrichments.")

    print("\n=== DECREASED GENES (excluding Gm*) ===")
    dec_named_results = run_go_enrichment(decreased_named, background=background)
    if not dec_named_results.empty:
        dec_named_results.to_csv(os.path.join(args.output_dir, "decreased_named_go_results.csv"), index=False)
        print(format_results(dec_named_results))
    else:
        print("No significant enrichments.")

    # Summary
    print("\n=== SUMMARY ===")
    for label, df in [("Increased", inc_results), ("Decreased", dec_results)]:
        if df.empty:
            print(f"{label}: no enrichments")
            continue
        for source in ["GO:CC", "GO:BP", "GO:MF", "KEGG"]:
            subset = df[df["source"] == source]
            if not subset.empty:
                top = subset.iloc[0]
                print(f"{label} top {source}: {top['name']} (p={top['p_value']:.2e}, n={top['intersection_size']})")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
