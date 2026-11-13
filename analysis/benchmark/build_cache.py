"""
Build the benchmark cache (one-time, ~10-20 min, < 16 GB RAM).

Reads the dense raw h5ad files block-wise with h5py, keeps only the rows listed
in data/split_manifest.tsv (v3 protocol, common cell types) and the Config-B
gene columns, and writes float32 raw-count matrices per modality x partition.

Usage:
    python -m analysis.benchmark.build_cache [--force]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.benchmark.data_split import build_cache, cache_exists, CACHE_DIR, DEFAULT_GENE_LIST


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default=CACHE_DIR)
    p.add_argument("--gene_list", default=DEFAULT_GENE_LIST)
    p.add_argument("--block_rows", type=int, default=4096)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if cache_exists(args.cache_dir) and not args.force:
        print(f"Cache already present in {args.cache_dir}; use --force to rebuild.")
        return
    info = build_cache(args.gene_list, args.cache_dir, args.block_rows)
    print(info)


if __name__ == "__main__":
    main()
