"""
CLI orchestration for the evaluation framework.

Usage:
    # Evaluate a single model (all metrics)
    python -m analysis.eval.runner --model pct5v2_cond --eval all

    # Evaluate specific metrics
    python -m analysis.eval.runner --model pct5v2_cond --eval ct_level,sample_level

    # Evaluate multiple models
    python -m analysis.eval.runner --model pct5v2_cond,pct5v2_cond_xr --eval all

    # Cross-model comparison
    python -m analysis.eval.runner --compare --models pct5v2_cond,pct5v2_cond_xr

    # Generate SLURM job script
    python -m analysis.eval.runner --model pct5v2_cond --eval all --slurm

    # List available models
    python -m analysis.eval.runner --list
"""

import argparse
import json
import os
import sys
import time

# Ensure project root on path
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_DIR)

from analysis.eval import registry
from analysis.eval.data import load_eval_data
from analysis.eval.inference import load_model, run_5source_inference, cleanup_model, get_imputed
from analysis.eval.report import format_summary_table, save_results, print_summary

EVAL_CATEGORIES = ['ct_level', 'sample_level', 'latent', 'gene_stratified']


def run_evaluation(model_name, eval_categories, output_dir):
    """Run evaluation for a single model.

    Parameters
    ----------
    model_name : str
    eval_categories : list of str (subset of EVAL_CATEGORIES)
    output_dir : str

    Returns
    -------
    dict of results
    """
    config = registry.get_model(model_name)

    # Skip if all requested categories already have results
    model_output_dir = os.path.join(output_dir, model_name)
    existing = [c for c in eval_categories
                if os.path.exists(os.path.join(model_output_dir, c, 'summary.json'))]
    if set(existing) >= set(eval_categories):
        print(f"\n  [skip] {model_name}: all requested evals already exist in {model_output_dir}")
        return None

    print(f"\n{'#'*60}")
    print(f"# Evaluating: {model_name}")
    print(f"# {config['description']}")
    run_label = os.path.basename(config['run_folder']) if config['run_folder'] else '(baseline)'
    print(f"# Run: {run_label}")
    print(f"{'#'*60}")

    # Load data (cached by gene list + normalization)
    normalization = config.get('normalization', 'log1p')
    eval_data = load_eval_data(config['gene_list'], normalization=normalization)

    # Load model and run inference
    print(f"\n  [inference] Loading model...")
    model, device = load_model(config)
    print(f"  [inference] Running 5-source inference...")
    t0 = time.time()
    sources = run_5source_inference(
        model, config, eval_data.X_cell, eval_data.X_nuc, device,
        ct_cell=eval_data.ct_cell, ct_nuc=eval_data.ct_nuc,
    )
    print(f"  [inference] Done in {time.time()-t0:.1f}s")

    results = {}

    # CT-level metrics
    if 'ct_level' in eval_categories:
        print(f"\n  [ct_level] Computing CT-level metrics...")
        from analysis.eval.metrics.ct_level import run_all_ct_level
        t0 = time.time()
        results['ct_level'] = run_all_ct_level(sources, eval_data, config)
        print(f"  [ct_level] Done in {time.time()-t0:.1f}s")

    # Sample-level metrics
    if 'sample_level' in eval_categories:
        print(f"\n  [sample_level] Computing sample-level metrics...")
        from analysis.eval.metrics.sample_level import run_all_sample_level
        t0 = time.time()
        results['sample_level'] = run_all_sample_level(sources, eval_data, config)
        print(f"  [sample_level] Done in {time.time()-t0:.1f}s")

    # Latent metrics (need model for z ablation — skip for baselines)
    if 'latent' in eval_categories and model is not None:
        print(f"\n  [latent] Computing latent metrics...")
        from analysis.eval.metrics.latent import run_all_latent
        t0 = time.time()
        results['latent'] = run_all_latent(sources, eval_data, model, config, device)
        print(f"  [latent] Done in {time.time()-t0:.1f}s")
    elif 'latent' in eval_categories:
        print(f"\n  [latent] Skipped (no model for baselines)")

    # Gene stratified (depends on ct_level + sample_level computations)
    if 'gene_stratified' in eval_categories:
        print(f"\n  [gene_stratified] Computing gene-stratified metrics...")
        from analysis.eval.metrics.gene_stratified import run_gene_stratified
        t0 = time.time()
        results['gene_stratified'] = run_gene_stratified(sources, eval_data, config)
        print(f"  [gene_stratified] Done in {time.time()-t0:.1f}s")

    # Cleanup
    cleanup_model(model)

    # Save results
    save_results(results, model_name, output_dir)
    print_summary(results, model_name)

    return results


def generate_slurm_script(model_names, eval_categories, output_dir):
    """Generate a SLURM job script for evaluation.

    Parameters
    ----------
    model_names : list of str
    eval_categories : list of str
    output_dir : str

    Returns
    -------
    str : path to generated script
    """
    models_str = ','.join(model_names)
    evals_str = ','.join(eval_categories)

    # Determine resource needs
    needs_gpu = any(
        registry.get_model(m)['model_class'] != 'baseline'
        for m in model_names
    )

    script = f"""#!/bin/bash
#SBATCH --job-name=eval_{model_names[0]}
#SBATCH --partition=pi_gerstein{",scavenge_gpu" if needs_gpu else ""}
#SBATCH --time=4:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
{"#SBATCH --gpus=1" if needs_gpu else ""}
#SBATCH --output=jobs/logs/slurm-%j_eval.out
#SBATCH --error=jobs/logs/slurm-%j_eval.err
#SBATCH --requeue

source ~/.bashrc
conda activate impute2

cd {_PROJECT_DIR}

python -m analysis.eval.runner \\
    --model {models_str} \\
    --eval {evals_str} \\
    --output_dir {output_dir}

echo "Evaluation complete: $(date)"
"""

    # Save script
    os.makedirs(os.path.join(_PROJECT_DIR, 'jobs', 'eval'), exist_ok=True)
    script_name = f"eval_{'_'.join(model_names)}.sh"
    script_path = os.path.join(_PROJECT_DIR, 'jobs', 'eval', script_name)
    with open(script_path, 'w') as f:
        f.write(script)

    print(f"SLURM script written to: {script_path}")
    print(f"Submit with: sbatch {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser(description='imputeVAE evaluation framework')
    parser.add_argument('--model', type=str, default=None,
                        help='Comma-separated model names (from registry)')
    parser.add_argument('--eval', type=str, default='all',
                        help='Comma-separated eval categories: '
                             'ct_level, sample_level, latent, gene_stratified, all')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: notes/eval_results/)')
    parser.add_argument('--compare', action='store_true',
                        help='Generate cross-model comparison table')
    parser.add_argument('--list', action='store_true',
                        help='List available models')
    parser.add_argument('--slurm', action='store_true',
                        help='Generate SLURM job script instead of running')
    args = parser.parse_args()

    # List models
    if args.list:
        print("\nAvailable models:")
        for name in registry.list_models():
            cfg = registry.get_model(name)
            run_label = os.path.basename(cfg['run_folder']) if cfg['run_folder'] else 'PENDING'
            norm = cfg.get('normalization', 'log1p')
            print(f"  {name:<20} {cfg['description']}")
            print(f"    {'':20} run: {run_label} | norm: {norm}")
            print(f"    {'':20} genes: {os.path.basename(cfg['gene_list'])}")
        return

    # Parse eval categories
    if args.eval == 'all':
        eval_cats = EVAL_CATEGORIES[:]
    else:
        eval_cats = [e.strip() for e in args.eval.split(',')]
        invalid = [e for e in eval_cats if e not in EVAL_CATEGORIES]
        if invalid:
            print(f"Error: Unknown eval categories: {invalid}")
            print(f"Available: {EVAL_CATEGORIES}")
            sys.exit(1)

    # Output directory
    output_dir = args.output_dir or os.path.join(_PROJECT_DIR, 'notes', 'eval_results')
    os.makedirs(output_dir, exist_ok=True)

    # Parse model names
    if args.model is None:
        if args.compare:
            model_names = registry.list_models()
        else:
            print("Error: --model is required (or use --list to see available models)")
            sys.exit(1)
    else:
        model_names = [m.strip() for m in args.model.split(',')]

    # Validate models
    for name in model_names:
        try:
            cfg = registry.get_model(name)
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
        if cfg['run_folder'] is None and cfg['model_class'] != 'baseline':
            print(f"Error: Model '{name}' has no run_folder yet (training still in progress)")
            sys.exit(1)

    # SLURM mode
    if args.slurm:
        generate_slurm_script(model_names, eval_cats, output_dir)
        return

    # Run evaluation
    all_results = {}
    for model_name in model_names:
        results = run_evaluation(model_name, eval_cats, output_dir)
        all_results[model_name] = results

    # Comparison table
    if len(model_names) > 1 or args.compare:
        print(f"\n{'='*60}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*60}")
        table = format_summary_table(all_results)
        print(table.to_string(index=False))
        table.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)
        print(f"\nComparison saved to: {output_dir}/comparison_table.csv")

    print(f"\nAll results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
