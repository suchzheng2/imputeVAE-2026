#!/usr/bin/env python3
"""
Epoch Dump Watchdog — process-and-delete evaluation.

Scans a run folder for epoch dumps, runs the full post-training evaluation
on each, saves results, and deletes the dump to free space.

Can be run once (default) or in a loop with --watch.

Usage:
    # One-shot: process all available dumps
    python eval_watchdog.py --run_folder <name>

    # Watch mode: poll every N minutes for new dumps
    python eval_watchdog.py --run_folder <name> --watch --interval 10

    # Specify analyses (default: correlation,heatmaps,gene_corr)
    python eval_watchdog.py --run_folder <name> --analyses correlation,heatmaps

    # Dry run: show what would be processed without doing it
    python eval_watchdog.py --run_folder <name> --dry_run
"""

import os
import sys
import gc
import json
import time
import argparse
import traceback
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from eval_utils import (
    find_run_folder, get_eval_dir, load_run_args,
    list_epoch_dumps, find_epoch_dump,
    load_common_filters, load_raw_data, print_memory_usage,
    is_stage3_run,
    RAW_CELL_PATH, RAW_NUCLEI_PATH,
)

AVAILABLE_ANALYSES = ['correlation', 'heatmaps', 'gene_corr', 'scib']


def get_processed_epochs(eval_dir):
    """Read the set of already-processed epochs from the tracker file."""
    tracker_path = os.path.join(eval_dir, 'watchdog_processed.json')
    if os.path.isfile(tracker_path):
        with open(tracker_path, 'r') as f:
            data = json.load(f)
        return set(data.get('processed_epochs', []))
    return set()


def mark_epoch_processed(eval_dir, epoch, results):
    """Record that an epoch has been fully processed."""
    tracker_path = os.path.join(eval_dir, 'watchdog_processed.json')
    if os.path.isfile(tracker_path):
        with open(tracker_path, 'r') as f:
            data = json.load(f)
    else:
        data = {'processed_epochs': [], 'details': {}}

    if epoch not in data['processed_epochs']:
        data['processed_epochs'].append(epoch)
        data['processed_epochs'].sort()
    data['details'][str(epoch)] = {
        'processed_at': datetime.now().isoformat(),
        'results': results,
    }

    with open(tracker_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_raw_data_once(run_args):
    """Load raw reference data (expensive, do once)."""
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

    return raw_cell, raw_nuclei


def process_epoch(run_folder, epoch, dump_path, analyses, is_stage3,
                  raw_cell, raw_nuclei, run_args):
    """Run all requested analyses on a single epoch dump."""
    eval_dir = get_eval_dir(run_folder)
    results = {'epoch': epoch, 'status': 'success', 'analyses': {}}

    use_log1p = run_args.get('raw_data_normalization', 'raw') == 'log1p'

    if 'heatmaps' in analyses:
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
            results['analyses']['heatmaps'] = r
        except Exception as e:
            print(f"    Heatmaps failed: {e}")
            traceback.print_exc()
            results['analyses']['heatmaps'] = {'status': 'failed', 'error': str(e)}

    if 'correlation' in analyses:
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
            results['analyses']['correlation'] = r
        except Exception as e:
            print(f"    Correlation failed: {e}")
            traceback.print_exc()
            results['analyses']['correlation'] = {'status': 'failed', 'error': str(e)}

    if 'gene_corr' in analyses:
        print(f"\n  [{epoch}] Running gene correlation...")
        try:
            from eval_gene_correlation import run_gene_correlation_analysis
            modality = _infer_modality(run_args)
            r = run_gene_correlation_analysis(
                run_folder, modality, epochs=[epoch],
                raw_cell_adata=raw_cell, raw_nuclei_adata=raw_nuclei,
            )
            results['analyses']['gene_corr'] = r
        except Exception as e:
            print(f"    Gene correlation failed: {e}")
            traceback.print_exc()
            results['analyses']['gene_corr'] = {'status': 'failed', 'error': str(e)}

    if 'scib' in analyses:
        print(f"\n  [{epoch}] Running scIB...")
        try:
            from evaluate import run_scib
            manifest = {'modality': _infer_modality(run_args)}
            run_scib(run_folder, [epoch], manifest)
            results['analyses']['scib'] = manifest.get('scib', {'status': 'unknown'})
        except Exception as e:
            print(f"    scIB failed: {e}")
            traceback.print_exc()
            results['analyses']['scib'] = {'status': 'failed', 'error': str(e)}

    gc.collect()
    return results


def _infer_modality(run_args):
    data_path = str(run_args.get('data_path', '')).lower()
    if 'nuclei' in data_path:
        return 'nuclei'
    return 'cell'


def run_watchdog(run_folder, analyses, watch=False, interval=10, dry_run=False):
    """Main watchdog loop."""
    run_args = load_run_args(run_folder)
    is_stage3 = is_stage3_run(run_args)
    eval_dir = get_eval_dir(run_folder)

    print(f"Run folder: {run_folder}")
    print(f"Stage3 mode: {is_stage3}")
    print(f"Analyses: {analyses}")
    if watch:
        print(f"Watch mode: polling every {interval} minutes")

    # Load raw data once (reused across all epochs)
    if not dry_run:
        raw_cell, raw_nuclei = load_raw_data_once(run_args)
    else:
        raw_cell = raw_nuclei = None

    while True:
        processed = get_processed_epochs(eval_dir)
        available = list_epoch_dumps(run_folder)
        new_epochs = [e for e in available if e not in processed]

        if new_epochs:
            print(f"\n{'='*60}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(new_epochs)} new epoch dumps: {new_epochs}")
            print(f"Already processed: {sorted(processed)}")

            for epoch in new_epochs:
                dump_path = find_epoch_dump(run_folder, epoch)
                if dump_path is None:
                    continue

                dump_size_mb = os.path.getsize(dump_path) / (1024 * 1024)
                print(f"\n--- Processing epoch {epoch} ({dump_size_mb:.0f} MB) ---")

                if dry_run:
                    print(f"  [DRY RUN] Would process and delete: {dump_path}")
                    continue

                # Process
                results = process_epoch(
                    run_folder, epoch, dump_path, analyses, is_stage3,
                    raw_cell, raw_nuclei, run_args,
                )

                # Record
                mark_epoch_processed(eval_dir, epoch, results)

                # Delete dump
                try:
                    os.remove(dump_path)
                    print(f"  Deleted dump: {dump_path} ({dump_size_mb:.0f} MB freed)")
                except OSError as e:
                    print(f"  Warning: could not delete {dump_path}: {e}")

                gc.collect()

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Batch complete.")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No new epoch dumps found.")

        if not watch:
            break

        print(f"Sleeping {interval} minutes...")
        time.sleep(interval * 60)

    # Cleanup
    if raw_cell is not None:
        del raw_cell, raw_nuclei
        gc.collect()

    print("\nWatchdog finished.")


def main():
    parser = argparse.ArgumentParser(description="Epoch Dump Watchdog")
    parser.add_argument("--run_folder", required=True, help="Run folder name or path")
    parser.add_argument("--analyses", default="correlation,heatmaps",
                        help=f"Comma-separated analyses (available: {AVAILABLE_ANALYSES})")
    parser.add_argument("--watch", action="store_true",
                        help="Watch mode: poll for new dumps repeatedly")
    parser.add_argument("--interval", type=int, default=10,
                        help="Minutes between polls in watch mode (default: 10)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be processed without doing it")
    args = parser.parse_args()

    run_folder = find_run_folder(args.run_folder)
    analyses = [a.strip() for a in args.analyses.split(',') if a.strip() in AVAILABLE_ANALYSES]
    if not analyses:
        print(f"No valid analyses specified. Available: {AVAILABLE_ANALYSES}")
        sys.exit(1)

    run_watchdog(run_folder, analyses, watch=args.watch,
                 interval=args.interval, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
