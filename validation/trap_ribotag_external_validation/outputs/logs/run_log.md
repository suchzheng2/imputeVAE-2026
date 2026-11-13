# Run log (2026-09-17/18)

## Environment

- conda env `impute2`, Python Python 3.12.2
- numpy 1.26.4
- pandas 1.5.3
- scipy 1.11.4
- sklearn 1.3.0
- matplotlib 3.8.0
- anndata 0.10.5.post1
- R 4.3.2 via module `R-bundle-Bioconductor/3.18-foss-2022b-R-4.3.2` (edgeR, limma; see outputs/logs/*sessionInfo.txt)
- seeds: 42 everywhere; final permutations/bootstraps = 1000 (development runs used 200)

## SLURM jobs (partition scavenge)

- 10905766: 01 model pseudobulk (3 min, 86 GB peak)
- 10905784: R smoke test
- 10905806: 05/06 limma-voom first run
- 10905946: 05/06 rerun with within-cell-type outlier rule
- 10905990: 09a GSE174303 limma-voom
- 10905994: 07+08 final statistics (1000 iterations, ~35 min)
- 10906009: 09 secondary datasets final (1000)

## Commands (from project root)

    python validation/trap_ribotag_external_validation/scripts/00_audit_and_download.py   # login node
    sbatch validation/trap_ribotag_external_validation/jobs/01_build_model_pseudobulk.sh
    python validation/trap_ribotag_external_validation/scripts/02_build_metadata.py
    python validation/trap_ribotag_external_validation/scripts/03_harmonize_genes.py
    python validation/trap_ribotag_external_validation/scripts/04_preprocess_gse210071.py
    sbatch validation/trap_ribotag_external_validation/jobs/05_06_preprocess_counts.sh
    sbatch validation/trap_ribotag_external_validation/jobs/09a_preprocess_gse174303.sh
    sbatch validation/trap_ribotag_external_validation/jobs/07_08_final.sh
    sbatch validation/trap_ribotag_external_validation/jobs/09_final.sh
    python validation/trap_ribotag_external_validation/scripts/10_make_figures.py
    python validation/trap_ribotag_external_validation/scripts/11_summarize_results.py > outputs/logs/11_summary.txt

## Warnings / notes

- `CondaError: Run 'conda init'` in SLURM stderr is benign (module init).
- pandas FutureWarnings suppressed in logs.
- Development runs of 07 with 200 permutations were superseded by the final SLURM run; concordance_summary_GSE210071.tsv from the first partial dev run was overwritten.

## Unresolved blockers

- None. GSE13379 IP/UB has n=1 unbound per population (descriptive only). GSE121199 D1 nuclear libraries are shallow (4/6 < 5M reads), which makes the D1 whole-cell-vs-nuclear contrast unreliable (reported, not excluded).
