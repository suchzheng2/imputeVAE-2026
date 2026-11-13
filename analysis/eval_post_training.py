#!/usr/bin/env python3
"""
Post-training evaluation on epoch dumps.

For each epoch dump in a run folder:
  1. Generate 3-panel UMAPs (cell_type, batch, modality) for encoded and decoded
  2. Run full evaluation (heatmaps, correlation, gene_corr)
  3. Run scIB metrics (via abc_env subprocess)

Usage:
    python eval_post_training.py --run_folder <name>
    python eval_post_training.py --run_folder <name> --epochs 0,50,100
    python eval_post_training.py --run_folder <name> --skip_umap
    python eval_post_training.py --run_folder <name> --skip_eval
    python eval_post_training.py --run_folder <name> --skip_scib
    python eval_post_training.py --run_folder <name> --delete_dumps
"""

import os
import sys
import gc
import json
import subprocess
import argparse
import traceback
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "imputeVAE"))

from eval_utils import (
    find_run_folder, get_eval_dir, load_run_args,
    list_epoch_dumps, find_epoch_dump,
    load_common_filters, load_raw_data, print_memory_usage,
    is_stage3_run,
    RAW_CELL_PATH, RAW_NUCLEI_PATH,
)


def run_umaps_for_epoch(run_folder, epoch, dump_path):
    """Generate 3-panel UMAPs (cell_type, batch, modality) for encoded and decoded."""
    from visualization import plot_umap_after_train_by_celltype_stage3

    print(f"\n  [{epoch}] Generating UMAPs...")
    for feature in ["encoded", "decoded"]:
        try:
            plot_umap_after_train_by_celltype_stage3(
                run_folder, feature=feature, data_split="train",
                epoch=epoch, file_path=dump_path,
            )
            print(f"    {feature} UMAP done")
        except Exception as e:
            print(f"    {feature} UMAP failed: {e}")
            traceback.print_exc()


def run_full_eval_for_epoch(run_folder, epoch, is_stage3, raw_cell, raw_nuclei, run_args):
    """Run full evaluation (heatmaps, correlation, gene_corr) for one epoch."""
    results = {}

    print(f"\n  [{epoch}] Running heatmaps...")
    try:
        if is_stage3:
            from eval_heatmaps import run_heatmap_analysis_stage3
            r = run_heatmap_analysis_stage3(
                run_folder, epoch,
                raw_cell_adata=raw_cell, raw_nuclei_adata=raw_nuclei,
            )
        else:
            from eval_heatmaps import run_heatmap_analysis
            modality = _infer_modality(run_args)
            r = run_heatmap_analysis(
                run_folder, modality, epoch,
                raw_cell_adata=raw_cell, raw_nuclei_adata=raw_nuclei,
            )
        results['heatmaps'] = r
        print(f"    Heatmaps: {r.get('status', 'unknown')}")
    except Exception as e:
        print(f"    Heatmaps failed: {e}")
        traceback.print_exc()
        results['heatmaps'] = {'status': 'failed', 'error': str(e)}

    print(f"\n  [{epoch}] Running correlation...")
    try:
        if is_stage3:
            from eval_correlation import run_correlation_and_shift_stage3
            r = run_correlation_and_shift_stage3(
                run_folder, epochs=[epoch],
                raw_cell_adata=raw_cell, raw_nuclei_adata=raw_nuclei,
            )
        else:
            from eval_correlation import run_correlation_and_shift
            modality = _infer_modality(run_args)
            r = run_correlation_and_shift(
                run_folder, modality, epochs=[epoch],
                raw_cell_adata=raw_cell, raw_nuclei_adata=raw_nuclei,
            )
        results['correlation'] = r
        print(f"    Correlation: {r.get('status', 'unknown')}")
    except Exception as e:
        print(f"    Correlation failed: {e}")
        traceback.print_exc()
        results['correlation'] = {'status': 'failed', 'error': str(e)}

    print(f"\n  [{epoch}] Running gene correlation...")
    try:
        from eval_gene_correlation import run_gene_correlation_analysis
        modality = _infer_modality(run_args)
        r = run_gene_correlation_analysis(
            run_folder, modality, epochs=[epoch],
            raw_cell_adata=raw_cell, raw_nuclei_adata=raw_nuclei,
        )
        results['gene_corr'] = r
        print(f"    Gene correlation: {r.get('status', 'unknown')}")
    except Exception as e:
        print(f"    Gene correlation failed: {e}")
        traceback.print_exc()
        results['gene_corr'] = {'status': 'failed', 'error': str(e)}

    gc.collect()
    return results


def run_scib_for_epoch(run_folder, epoch, is_stage3, run_args):
    """Run scIB metrics via subprocess in abc_env."""
    print(f"\n  [{epoch}] Running scIB metrics...")

    # Find conda
    conda_path = None
    for candidate in ['conda', os.path.expanduser('~/miniconda3/bin/conda'),
                      os.path.expanduser('~/anaconda3/bin/conda')]:
        try:
            subprocess.run([candidate, '--version'], capture_output=True, check=True)
            conda_path = candidate
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    if conda_path is None:
        print("    Warning: conda not found, skipping scIB")
        return {'status': 'skipped', 'reason': 'conda not found'}

    # Check --no-banner support
    supports_no_banner = False
    try:
        r = subprocess.run([conda_path, 'run', '-h'], capture_output=True, text=True)
        if '--no-banner' in r.stdout:
            supports_no_banner = True
    except Exception:
        pass

    eval_scib_path = os.path.join(_SCRIPT_DIR, 'eval_scib.py')
    scib_mode = 'stage3_full' if is_stage3 else 'single'

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
        modality = _infer_modality(run_args)
        cmd.extend(['--modality', modality])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode == 0:
            results_path = os.path.join(
                run_folder, 'evaluation', 'scib_metrics', f'epoch_{epoch}', 'scib_results.json'
            )
            if os.path.isfile(results_path):
                with open(results_path, 'r') as f:
                    result = json.load(f)
                print(f"    scIB: success")
                return result
            else:
                print(f"    scIB: success (results file not found)")
                return {'status': 'success', 'note': 'results file not found'}
        else:
            print(f"    scIB: failed (exit code {proc.returncode})")
            if proc.stderr:
                print(f"    stderr: {proc.stderr[:500]}")
            return {'status': 'failed', 'stderr': proc.stderr[:500]}
    except subprocess.TimeoutExpired:
        print(f"    scIB: timed out (30min)")
        return {'status': 'failed', 'reason': 'timeout'}
    except Exception as e:
        print(f"    scIB: error: {e}")
        return {'status': 'failed', 'reason': str(e)}


def _infer_modality(run_args):
    data_path = str(run_args.get('data_path', '')).lower()
    if 'nuclei' in data_path:
        return 'nuclei'
    return 'cell'


def main():
    parser = argparse.ArgumentParser(description="Post-training evaluation on epoch dumps")
    parser.add_argument("--run_folder", required=True, help="Run folder name or path")
    parser.add_argument("--epochs", type=str, default=None,
                        help="Comma-separated epochs (default: all available)")
    parser.add_argument("--skip_umap", action="store_true",
                        help="Skip UMAP generation")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip full evaluation (heatmaps, correlation, gene_corr)")
    parser.add_argument("--skip_scib", action="store_true",
                        help="Skip scIB metrics")
    parser.add_argument("--delete_dumps", action="store_true",
                        help="Delete epoch dumps after processing")
    args = parser.parse_args()

    run_folder = find_run_folder(args.run_folder)
    run_args = load_run_args(run_folder)
    is_stage3 = is_stage3_run(run_args)

    available_epochs = list_epoch_dumps(run_folder)
    if args.epochs:
        selected = set(int(e) for e in args.epochs.split(','))
        available_epochs = [e for e in available_epochs if e in selected]

    if not available_epochs:
        print("No epoch dumps found.")
        return

    print(f"Run folder: {run_folder}")
    print(f"Stage3 mode: {is_stage3}")
    print(f"Epochs to process: {available_epochs}")
    print(f"UMAP: {'skip' if args.skip_umap else 'yes'}")
    print(f"Full eval: {'skip' if args.skip_eval else 'yes'}")
    print(f"scIB: {'skip' if args.skip_scib else 'yes'}")
    print(f"Delete dumps after: {'yes' if args.delete_dumps else 'no'}")

    # Load raw data once if needed for eval
    raw_cell = raw_nuclei = None
    if not args.skip_eval:
        det_rate = float(run_args.get('detection_rate_threshold', 1.0))
        exclude_mito = run_args.get('exclude_mitocarta', True)
        protocol = run_args.get('protocol', None)
        gene_list_path = run_args.get('gene_list', None)
        if gene_list_path and not os.path.isabs(gene_list_path):
            _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            gene_list_path = os.path.join(_project_dir, gene_list_path)
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

    # Process each epoch
    manifest = {'run_folder': run_folder, 'started_at': datetime.now().isoformat(), 'epochs': {}}

    for epoch in available_epochs:
        dump_path = find_epoch_dump(run_folder, epoch)
        if dump_path is None:
            print(f"\n  [{epoch}] Dump not found, skipping")
            continue

        dump_size_mb = os.path.getsize(dump_path) / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"Processing epoch {epoch} ({dump_size_mb:.0f} MB)")

        epoch_results = {}

        # 1. UMAPs
        if not args.skip_umap:
            run_umaps_for_epoch(run_folder, epoch, dump_path)
            epoch_results['umap'] = {'status': 'success'}

        # 2. Full evaluation
        if not args.skip_eval:
            eval_results = run_full_eval_for_epoch(
                run_folder, epoch, is_stage3, raw_cell, raw_nuclei, run_args,
            )
            epoch_results['eval'] = eval_results

        # 3. scIB metrics
        if not args.skip_scib:
            scib_result = run_scib_for_epoch(run_folder, epoch, is_stage3, run_args)
            epoch_results['scib'] = scib_result

        manifest['epochs'][str(epoch)] = epoch_results

        # 3. Delete dump if requested
        if args.delete_dumps:
            try:
                os.remove(dump_path)
                print(f"  Deleted dump: {dump_path} ({dump_size_mb:.0f} MB freed)")
            except OSError as e:
                print(f"  Warning: could not delete {dump_path}: {e}")

        gc.collect()

    # Save manifest
    manifest['completed_at'] = datetime.now().isoformat()
    eval_dir = get_eval_dir(run_folder)
    manifest_path = os.path.join(eval_dir, 'post_training_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Done. Processed {len(available_epochs)} epochs.")
    print(f"Manifest: {manifest_path}")

    # Cleanup
    if raw_cell is not None:
        del raw_cell, raw_nuclei
        gc.collect()


if __name__ == '__main__':
    main()
