#!/usr/bin/env python3
"""
Unified Evaluation Pipeline -- Orchestrator.

Dispatches to modular analysis modules, loads shared data once,
and writes manifest.json + summary.txt.

Usage:
    python evaluate.py --run_folder <name> --modality <cell|nuclei> \
        --analyses loss,scale,correlation,gene_corr,shift,scib,heatmaps
    python evaluate.py --run_folder <name> --modality nuclei --analyses all
"""

import os
import sys
import gc
import json
import time
import argparse
import subprocess
import traceback
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from eval_utils import (
    find_run_folder, get_eval_dir, load_run_args, list_epoch_dumps,
    load_common_filters, load_raw_data, print_memory_usage,
    is_stage3_run,
    RAW_CELL_PATH, RAW_NUCLEI_PATH,
)

ALL_ANALYSES = ['loss', 'scale', 'correlation', 'gene_corr', 'shift', 'scib', 'heatmaps']

# Analyses that need raw H5AD data loaded
NEEDS_RAW_DATA = {'correlation', 'gene_corr', 'shift', 'heatmaps'}


def parse_analyses(analyses_str):
    """Parse comma-separated analyses string."""
    if analyses_str.strip().lower() == 'all':
        return list(ALL_ANALYSES)
    requested = [a.strip().lower() for a in analyses_str.split(',')]
    invalid = [a for a in requested if a not in ALL_ANALYSES]
    if invalid:
        print(f"Warning: Unknown analyses ignored: {invalid}")
    return [a for a in requested if a in ALL_ANALYSES]


def load_raw_data_if_needed(analyses, run_args):
    """Load raw H5AD data only if at least one analysis needs it."""
    if not NEEDS_RAW_DATA.intersection(analyses):
        return None, None

    det_rate = float(run_args.get('detection_rate_threshold', 1.0))
    exclude_mito = run_args.get('exclude_mitocarta', True)
    protocol = run_args.get('protocol', None)
    gene_list_path = run_args.get('gene_list', None)

    # Resolve gene_list path relative to project root if needed
    if gene_list_path and not os.path.isabs(gene_list_path):
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gene_list_path = os.path.join(project_dir, gene_list_path)

    common_genes, common_cell_types, mitocarta_genes = load_common_filters()

    print("\n=== Loading raw reference data ===")
    raw_cell = load_raw_data(RAW_CELL_PATH, common_genes, common_cell_types,
                             mitocarta_genes, det_rate, exclude_mito,
                             protocol=protocol, batch='c3A',
                             gene_list=gene_list_path)
    print_memory_usage("After raw cell load")

    raw_nuclei = load_raw_data(RAW_NUCLEI_PATH, common_genes, common_cell_types,
                               mitocarta_genes, det_rate, exclude_mito,
                               protocol=protocol, batch='n3A',
                               gene_list=gene_list_path)
    print_memory_usage("After raw nuclei load")

    return raw_cell, raw_nuclei


def run_loss(run_folder, manifest):
    """Run loss curve analysis."""
    from eval_loss import run_loss_analysis
    print("\n=== Loss Curve Analysis ===")
    result = run_loss_analysis(run_folder)
    manifest['loss'] = result
    return result


def run_scale(run_folder, epochs, manifest):
    """Run decoder scale analysis."""
    from eval_decoder_scale import run_decoder_scale_analysis
    print("\n=== Decoder Scale Analysis ===")
    result = run_decoder_scale_analysis(run_folder, epochs=epochs)
    manifest['scale'] = result
    return result


def run_correlation(run_folder, modality, epochs, raw_cell, raw_nuclei, manifest):
    """Run correlation + shift analysis."""
    run_args = load_run_args(run_folder)
    is_stage3 = is_stage3_run(run_args)

    if is_stage3:
        from eval_correlation import run_correlation_and_shift_stage3
        print("\n=== 4x4 Correlation & Shift Analysis (stage3_v2) ===")
        result = run_correlation_and_shift_stage3(
            run_folder, epochs=epochs,
            raw_cell_adata=raw_cell, raw_nuclei_adata=raw_nuclei,
        )
    else:
        from eval_correlation import run_correlation_and_shift
        print("\n=== Correlation & Shift Analysis ===")
        result = run_correlation_and_shift(
            run_folder, modality, epochs=epochs,
            raw_cell_adata=raw_cell, raw_nuclei_adata=raw_nuclei,
        )
    manifest['correlation'] = result
    # shift is embedded in correlation module
    manifest['shift'] = {'status': result.get('status', 'skipped'), 'via': 'correlation'}
    return result


def run_gene_corr(run_folder, modality, epochs, raw_cell, raw_nuclei, manifest):
    """Run gene-level correlation analysis."""
    from eval_gene_correlation import run_gene_correlation_analysis
    print("\n=== Gene-Level Correlation Analysis ===")
    result = run_gene_correlation_analysis(
        run_folder, modality, epochs=epochs,
        raw_cell_adata=raw_cell, raw_nuclei_adata=raw_nuclei,
    )
    manifest['gene_corr'] = result
    return result


def run_scib(run_folder, epochs, manifest):
    """Run scib metrics via subprocess in abc_env."""
    print("\n=== scib Metrics (subprocess: abc_env) ===")

    # Check if conda is available
    conda_path = None
    for candidate in ['conda', os.path.expanduser('~/miniconda3/bin/conda'),
                      os.path.expanduser('~/anaconda3/bin/conda')]:
        try:
            subprocess.run([candidate, '--version'], capture_output=True, timeout=10)
            conda_path = candidate
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if conda_path is None:
        print("  Warning: conda not found, skipping scib analysis")
        manifest['scib'] = {'status': 'skipped', 'reason': 'conda not found'}
        return

    try:
        conda_run_help = subprocess.run(
            [conda_path, 'run', '-h'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        conda_run_help_text = f"{conda_run_help.stdout}\n{conda_run_help.stderr}"
    except Exception:
        conda_run_help_text = ""

    supports_no_banner = '--no-banner' in conda_run_help_text

    eval_scib_path = os.path.join(_SCRIPT_DIR, 'eval_scib.py')
    scib_results = {}

    run_args = load_run_args(run_folder)
    is_stage3 = is_stage3_run(run_args)
    scib_mode = 'stage3_full' if is_stage3 else 'single'

    for epoch in epochs:
        print(f"  Running scib for epoch {epoch} (mode={scib_mode})...")
        cmd = [conda_path, 'run', '-n', 'abc_env']
        if supports_no_banner:
            cmd.append('--no-banner')
        cmd.extend([
            'python', eval_scib_path,
            '--run_folder', run_folder,
            '--epoch', str(epoch),
            '--mode', scib_mode,
        ])
        if not is_stage3:
            cmd.extend(['--modality', manifest['modality']])
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if proc.returncode == 0:
                print(f"    Epoch {epoch}: success")
                # Try to read the results JSON
                results_path = os.path.join(
                    run_folder, 'evaluation', 'scib_metrics', f'epoch_{epoch}', 'scib_results.json'
                )
                if os.path.isfile(results_path):
                    with open(results_path, 'r') as f:
                        scib_results[epoch] = json.load(f)
                else:
                    scib_results[epoch] = {'status': 'success', 'note': 'results file not found'}
            else:
                print(f"    Epoch {epoch}: failed (exit code {proc.returncode})")
                if proc.stderr:
                    print(f"    stderr: {proc.stderr[:500]}")
                scib_results[epoch] = {'status': 'failed', 'stderr': proc.stderr[:500]}
        except subprocess.TimeoutExpired:
            print(f"    Epoch {epoch}: timed out (30min)")
            scib_results[epoch] = {'status': 'failed', 'reason': 'timeout'}
        except Exception as e:
            print(f"    Epoch {epoch}: error: {e}")
            scib_results[epoch] = {'status': 'failed', 'reason': str(e)}

    trend_plot = plot_scib_trends(run_folder, scib_results)

    manifest['scib'] = {
        'status': 'success' if any(r.get('status') == 'success' for r in scib_results.values()) else 'failed',
        'epochs': scib_results,
        'trend_plot': trend_plot,
    }


def plot_scib_trends(run_folder, scib_results):
    """Plot scIB metric trends over epochs and save a companion CSV."""
    valid_epochs = sorted(
        int(epoch)
        for epoch, result in scib_results.items()
        if isinstance(result, dict) and result.get('status') == 'success'
    )
    if not valid_epochs:
        return None

    trend_rows = []

    for epoch in valid_epochs:
        result = scib_results[epoch]
        encoded = result.get('encoded', {})
        decoded = result.get('decoded', {})
        trend_rows.append({
            'epoch': epoch,
            'encoded_ASW_batch': encoded.get('ASW_batch'),
            'decoded_ASW_batch': decoded.get('ASW_batch'),
            'delta_ASW_batch': (
                None
                if encoded.get('ASW_batch') is None or decoded.get('ASW_batch') is None
                else decoded.get('ASW_batch') - encoded.get('ASW_batch')
            ),
            'encoded_iLISI_batch': encoded.get('iLISI_batch'),
            'decoded_iLISI_batch': decoded.get('iLISI_batch'),
            'delta_iLISI_batch': (
                None
                if encoded.get('iLISI_batch') is None or decoded.get('iLISI_batch') is None
                else decoded.get('iLISI_batch') - encoded.get('iLISI_batch')
            ),
            'encoded_cLISI_cell_type': encoded.get('cLISI_cell_type'),
            'decoded_cLISI_cell_type': decoded.get('cLISI_cell_type'),
            'delta_cLISI_cell_type': (
                None
                if encoded.get('cLISI_cell_type') is None or decoded.get('cLISI_cell_type') is None
                else decoded.get('cLISI_cell_type') - encoded.get('cLISI_cell_type')
            ),
        })

    output_dir = os.path.join(run_folder, 'evaluation', 'scib_metrics')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'scib_trends.png')
    csv_path = os.path.join(output_dir, 'scib_trends.csv')

    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    metric_specs = [
        ('ASW_batch', 'ASW Batch', axes[0]),
        ('iLISI_batch', 'iLISI Batch', axes[1]),
        ('cLISI_cell_type', 'cLISI Cell Type', axes[2]),
    ]

    for metric_key, title, ax in metric_specs:
        enc_col = f'encoded_{metric_key}'
        dec_col = f'decoded_{metric_key}'
        delta_col = f'delta_{metric_key}'

        ax.plot(
            trend_df['epoch'],
            trend_df[enc_col],
            marker='o',
            label='encoded',
            color='#1f77b4',
        )
        ax.plot(
            trend_df['epoch'],
            trend_df[dec_col],
            marker='o',
            label='decoded',
            color='#ff7f0e',
        )
        ax.plot(
            trend_df['epoch'],
            trend_df[delta_col],
            marker='x',
            linestyle='--',
            label='decoded - encoded',
            color='#2ca02c',
        )
        ax.set_title(f'{title} Over Epochs')
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel('Epoch')

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  scib trend plot saved to: {plot_path}")
    print(f"  scib trend table saved to: {csv_path}")
    return plot_path


def run_heatmaps(run_folder, modality, epochs, raw_cell, raw_nuclei, manifest):
    """Run heatmap analysis."""
    run_args = load_run_args(run_folder)
    is_stage3 = is_stage3_run(run_args)

    if is_stage3:
        from eval_heatmaps import run_heatmap_analysis_stage3
        print("\n=== 4-Source Heatmap Analysis (stage3_v2) ===")
    else:
        from eval_heatmaps import run_heatmap_analysis
        print("\n=== Heatmap Analysis ===")

    # Only run for last epoch by default (heatmaps are expensive)
    heatmap_epochs = [epochs[-1]] if epochs else []
    results = {}
    for epoch in heatmap_epochs:
        print(f"  Epoch {epoch}:")
        if is_stage3:
            result = run_heatmap_analysis_stage3(
                run_folder, epoch,
                raw_cell_adata=raw_cell, raw_nuclei_adata=raw_nuclei,
            )
        else:
            result = run_heatmap_analysis(
                run_folder, modality, epoch,
                raw_cell_adata=raw_cell, raw_nuclei_adata=raw_nuclei,
            )
        results[epoch] = result

    manifest['heatmaps'] = {
        'status': 'success' if results else 'skipped',
        'epochs': {str(k): v for k, v in results.items()},
    }


def write_manifest(eval_dir, manifest):
    """Write manifest.json."""
    manifest_path = os.path.join(eval_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest written to: {manifest_path}")


def write_summary(eval_dir, manifest, run_folder, modality, analyses):
    """Write a unified text summary."""
    lines = []
    lines.append(f"# Unified Evaluation Summary")
    lines.append(f"Run: {os.path.basename(run_folder)}")
    lines.append(f"Modality: {modality}")
    lines.append(f"Analyses: {', '.join(analyses)}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for analysis in analyses:
        result = manifest.get(analysis, {})
        status = result.get('status', 'not run')
        lines.append(f"## {analysis}: {status}")

        if analysis == 'loss':
            if 'train_epochs' in result:
                lines.append(f"  Train epochs: {result['train_epochs']}")
            if 'valid_epochs' in result:
                lines.append(f"  Valid epochs: {result['valid_epochs']}")

        elif analysis == 'scale':
            if 'final_mean' in result:
                lines.append(f"  Final mean: {result['final_mean']:.4f}")
                lines.append(f"  Final std: {result['final_std']:.4f}")

        elif analysis in ('correlation', 'shift'):
            if 'final_mean_shift' in result:
                lines.append(f"  Final mean shift: {result['final_mean_shift']:.4f}")
            if 'epochs_analyzed' in result:
                lines.append(f"  Epochs analyzed: {result['epochs_analyzed']}")

        elif analysis == 'scib':
            epochs_data = result.get('epochs', {})
            for ep, ep_result in epochs_data.items():
                lines.append(f"  Epoch {ep}: {ep_result.get('status', 'unknown')}")

        lines.append("")

    summary_path = os.path.join(eval_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Summary written to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Pipeline")
    parser.add_argument("--run_folder", required=True, help="Run folder name or path")
    parser.add_argument("--modality", default="nuclei", choices=['cell', 'nuclei'],
                        help="Modality (default: nuclei)")
    parser.add_argument("--analyses", default="all",
                        help="Comma-separated analyses or 'all' (default: all)")
    parser.add_argument("--epochs", type=str, default=None,
                        help="Comma-separated epochs (default: all available)")
    parser.add_argument("--heatmap_epochs", type=str, default=None,
                        help="Epochs for heatmaps (default: last epoch only)")
    args = parser.parse_args()

    run_folder = find_run_folder(args.run_folder)
    analyses = parse_analyses(args.analyses)
    run_args = load_run_args(run_folder)

    # Determine epochs
    available_epochs = list_epoch_dumps(run_folder)
    if args.epochs:
        epochs = sorted([int(e) for e in args.epochs.split(',')])
        epochs = [e for e in epochs if e in available_epochs]
    else:
        epochs = available_epochs

    print(f"Run folder: {run_folder}")
    print(f"Modality: {args.modality}")
    print(f"Analyses: {analyses}")
    print(f"Available epochs: {available_epochs}")
    print(f"Selected epochs: {epochs}")

    eval_dir = get_eval_dir(run_folder)
    manifest = {
        'run_folder': run_folder,
        'modality': args.modality,
        'analyses_requested': analyses,
        'epochs': epochs,
        'started_at': datetime.now().isoformat(),
    }

    t_start = time.time()

    # Load raw data once if needed
    raw_cell, raw_nuclei = load_raw_data_if_needed(analyses, run_args)

    # Dispatch analyses
    for analysis in analyses:
        t_analysis = time.time()
        try:
            if analysis == 'loss':
                run_loss(run_folder, manifest)

            elif analysis == 'scale':
                run_scale(run_folder, epochs, manifest)

            elif analysis in ('correlation', 'shift'):
                # shift is bundled with correlation
                if 'correlation' not in manifest:
                    run_correlation(run_folder, args.modality, epochs,
                                    raw_cell, raw_nuclei, manifest)

            elif analysis == 'gene_corr':
                run_gene_corr(run_folder, args.modality, epochs,
                              raw_cell, raw_nuclei, manifest)

            elif analysis == 'scib':
                scib_epochs = epochs if epochs else available_epochs
                run_scib(run_folder, scib_epochs, manifest)

            elif analysis == 'heatmaps':
                heatmap_epochs = epochs
                if args.heatmap_epochs:
                    heatmap_epochs = sorted([int(e) for e in args.heatmap_epochs.split(',')])
                run_heatmaps(run_folder, args.modality, heatmap_epochs,
                             raw_cell, raw_nuclei, manifest)

            elapsed = time.time() - t_analysis
            print(f"  [{analysis}] completed in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t_analysis
            print(f"  [{analysis}] FAILED after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            manifest[analysis] = {'status': 'failed', 'error': str(e)}

    # Cleanup
    del raw_cell, raw_nuclei
    gc.collect()

    # Finalize
    manifest['completed_at'] = datetime.now().isoformat()
    manifest['total_time_seconds'] = round(time.time() - t_start, 1)

    write_manifest(eval_dir, manifest)
    write_summary(eval_dir, manifest, run_folder, args.modality, analyses)

    print(f"\nTotal time: {manifest['total_time_seconds']:.1f}s")
    print(f"Results in: {eval_dir}")


if __name__ == '__main__':
    main()
