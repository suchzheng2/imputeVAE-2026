# Experiment History (detailed model/job records)

## Current Best Models (Config B, 10,134 genes)
- **MSE cond+xr**: `run/run_2026-04-02-10-51-32_3738876` — classif 95.8%, DE +5.1%
- **MSE cond**: `run/run_2026-04-02-02-02-35_3345839` — var ratio 11.5%
- **MSE shared**: `run/run_2026-04-01-14-01-16_1946018` — baseline

## PerCT_5pct Models (16,439 genes)
- Stage 1: `run/run_2026-04-02-19-27-35_3767601`
- Stage 2 cond: `run/run_2026-04-07-02-12-05_3779214`
- Stage 2 cond+xr: `run/run_2026-04-07-02-12-05_3551465`
- Key results: cond+xr DE overlap 54.7%, gene corr 0.963, classif 94.2%, var 0.035

## Stage 1 Normalized Loss Probes (2026-05-01)
- Winner: job 4153808, beta=0.1, no anneal → recon=0.251, JSD=0.08
- Run: `run/run_2026-05-01-12-01-55_1984590`

## CPM Experiment (2026-04-16 through 2026-05-13)
- Stage 1 (normloss): job 4521167, run `run/run_2026-05-08-13-22-30_3907651`
- Stage 2 cond: job 4598108, run `run/run_2026-05-11-04-43-20_620735` (gene corr 0.888, var 0.085)
- Stage 2 cond+xr: job 4598109, run `run/run_2026-05-11-19-20-59_1045978` (gene corr 0.783)
- Original CPM had JSD stuck at ln(2) — normloss fixed it

## Conditioning + CPM Full Results (2026-05-12)

| Job | Config | Gene corr | Shift | R² |
|-----|--------|-----------|-------|----|
| 4514645 | log1p, learned emb 8d, dec-only | 0.954 | +0.195 | 0.768 |
| 4607333 | log1p, single-stage cVAE | 0.932 | +0.211 | 0.792 |
| 4598336 | log1p, scalar, enc+dec (2-stage) | 0.919 | +0.209 | 0.494 |
| 4598175 | log1p, scalar, dec-only | 0.902 | +0.216 | 0.097 |
| 4598108 | CPM, learned emb 8d, dec-only | 0.888 | +0.190 | 0.475 |
| 4598109 | CPM, learned emb+xr, dec-only | 0.783 | +0.150 | 0.314 |
| 4598176 | CPM, scalar, dec-only | 0.796 | -0.175 | -0.855 |
| 4598337 | CPM, scalar, enc+dec (2-stage) | ? | ? | ? |
| 4607334 | CPM, single-stage cVAE | ? | ? | ? |

### Run folders
- 4514645: DELETED (normloss bug, retrain as job 5570485)
- 4607333: `run/run_2026-05-11-19-52-25_134850`
- 4598336: `run/run_2026-05-11-06-59-55_3983979`
- 4598175: `run/run_2026-05-11-06-02-02_625347`
- 4598108: `run/run_2026-05-11-04-43-20_620735`
- 4598109: `run/run_2026-05-11-19-20-59_1045978`
- 4598176: `run/run_2026-05-11-19-20-59_2898108`
- 4598337: `run/run_2026-05-12-10-47-50_652740`
- 4607334: `run/run_2026-05-12-15-07-59_1408472`

## Within-CT Variance Ratios (median, cross-decode)
| Model | Norm | Mean | Median |
|-------|------|------|--------|
| log1p, dec-only, learned 8d (4514645) | log1p | 0.397 | 0.454 |
| log1p, single-stage cVAE (4607333) | log1p | 0.360 | 0.402 |
| CPM, dec-only, learned 8d (4598108) | CPM | 0.111 | 0.085 |

## Factorial Experiment (2026-05-28)
- 16 runs: norm(log1p/cpm) × cond(emb8/scalar) × target(dec/encdec) × stage(2stage/1stage)
- Fixed: ConfigB, vae_js, latent 32, beta 0.1, normloss ON, 20% subset, v3, no XR
- Scripts: `jobs/factorial/f{01-16}_*.sh`
- S1 models: log1p=`run_2026-05-01-12-01-55_1984590`, cpm=`run_2026-05-08-13-22-30_3907651`
