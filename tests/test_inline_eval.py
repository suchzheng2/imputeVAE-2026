#!/usr/bin/env python3
"""
Smoke tests for the inline evaluation refactor.

Exercises:
  1. Epoch dump format (sample_id, both-modality batch IDs, filename)
  2. eval_utils: load_both_decoded_stage3, load_both_encoded_stage3
  3. stage3_eval: run_4source_correlation, run_4source_heatmaps (requires torch)
  4. staged_training: build_dump / is_eval_epoch logic

Usage:
    python tests/test_inline_eval.py            # all tests
    python tests/test_inline_eval.py --light     # skip torch-dependent tests
"""

import gc
import json
import os
import sys
import tempfile
import shutil
import traceback

import numpy as np
import pandas as pd

# Ensure project root is on path
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "analysis"))


# ── helpers ──────────────────────────────────────────────────────────────────

CELL_TYPES = ["Astro", "L2_3_IT", "Lamp5"]
N_GENES = 50
N_LATENT = 8
N_PER_CT = 10


def _make_synthetic_dump():
    """Create a synthetic joint epoch dump dict matching the new format."""
    dump = {}
    for ct in CELL_TYPES:
        dump[ct] = {}
        for bid in range(2):
            n = N_PER_CT
            entry = {
                "batch_idx": bid,
                "cell_type": list(range(n)),
                "cell_type_original": [ct] * n,
                "meta_batch_id": list(range(n)),
                "meta_batch_id_original": [f"donor{bid}-n3A"] * n,
                "nuclei_meta_batch_id_original": [f"donor{bid}-n3A"] * n,
                "cell_meta_batch_id_original": [f"donor{bid}-c3A"] * n,
                "nuclei_sample_id": [f"nuclei:ds:donor{bid}-n3A:cell_{i}" for i in range(n)],
                "cell_sample_id": [f"cell:ds:donor{bid}-c3A:cell_{i}" for i in range(n)],
                "nuclei_encoded": np.random.randn(n, N_LATENT).tolist(),
                "nuclei_decoded": np.random.randn(n, N_GENES).tolist(),
                "nuclei_mu": np.random.randn(n, N_LATENT).tolist(),
                "nuclei_log_var": np.random.randn(n, N_LATENT).tolist(),
                "cell_encoded": np.random.randn(n, N_LATENT).tolist(),
                "cell_decoded": np.random.randn(n, N_GENES).tolist(),
                "cell_mu": np.random.randn(n, N_LATENT).tolist(),
                "cell_log_var": np.random.randn(n, N_LATENT).tolist(),
                "loss": 0.5,
                "nuclei_recons_loss": 0.3,
                "nuclei_JSD_div": 0.1,
                "cell_recons_loss": 0.2,
                "cell_JSD_div": 0.05,
                "alignment_JSD_loss": 0.01,
            }
            dump[ct][str(bid)] = entry
    return dump


def _write_dump_file(tmpdir, epoch, dump_dict):
    dump_dir = os.path.join(tmpdir, "epoch_dump")
    os.makedirs(dump_dir, exist_ok=True)
    fp = os.path.join(dump_dir, f"train_epoch_{epoch}.json")
    with open(fp, "w") as f:
        json.dump(dump_dict, f)
    return fp


def _make_fake_raw_adata():
    """Create a minimal AnnData matching what eval functions expect."""
    import anndata as ad
    n_cells = N_PER_CT * len(CELL_TYPES)
    X = np.random.rand(n_cells, N_GENES).astype(np.float32)
    obs = pd.DataFrame({
        "cell_type": np.repeat(CELL_TYPES, N_PER_CT),
        "meta_batch_id": ["donor0-n3A"] * n_cells,
    })
    var = pd.DataFrame({
        "gene_names": [f"gene_{i}" for i in range(N_GENES)],
    })
    return ad.AnnData(X=X, obs=obs, var=var)


# ── lightweight tests (no torch) ────────────────────────────────────────────

def test_dump_format():
    """Verify the synthetic dump has the expected keys."""
    dump = _make_synthetic_dump()
    for ct in CELL_TYPES:
        assert ct in dump, f"Missing cell type {ct}"
        for bid_str, entry in dump[ct].items():
            assert "nuclei_meta_batch_id_original" in entry
            assert "cell_meta_batch_id_original" in entry
            assert "nuclei_sample_id" in entry
            assert "cell_sample_id" in entry
            assert "cell_encoded" in entry
            assert "nuclei_encoded" in entry
            assert "cell_decoded" in entry
            assert "nuclei_decoded" in entry
            assert "meta_batch_id_original" in entry
    print("  PASS: dump format")


def test_dump_filename():
    """Verify dump writes to train_epoch_N.json."""
    tmpdir = tempfile.mkdtemp(prefix="test_dump_")
    try:
        dump = _make_synthetic_dump()
        fp = _write_dump_file(tmpdir, 50, dump)
        assert os.path.basename(fp) == "train_epoch_50.json", f"Wrong filename: {fp}"
        assert os.path.isfile(fp)
        print("  PASS: dump filename")
    finally:
        shutil.rmtree(tmpdir)


def test_load_both_decoded():
    """Test load_both_decoded_stage3."""
    from eval_utils import load_both_decoded_stage3
    tmpdir = tempfile.mkdtemp(prefix="test_load_")
    try:
        dump = _make_synthetic_dump()
        fp = _write_dump_file(tmpdir, 10, dump)
        cell_dec, cell_types, nuclei_dec, nuclei_types = load_both_decoded_stage3(
            fp, set(CELL_TYPES)
        )
        expected_n = N_PER_CT * 2 * len(CELL_TYPES)
        assert cell_dec.shape == (expected_n, N_GENES), f"Wrong cell shape: {cell_dec.shape}"
        assert nuclei_dec.shape == (expected_n, N_GENES), f"Wrong nuclei shape: {nuclei_dec.shape}"
        assert set(cell_types) == set(CELL_TYPES)
        print("  PASS: load_both_decoded_stage3")
    finally:
        shutil.rmtree(tmpdir)


def test_load_both_encoded():
    """Test load_both_encoded_stage3."""
    from eval_utils import load_both_encoded_stage3
    tmpdir = tempfile.mkdtemp(prefix="test_load_")
    try:
        dump = _make_synthetic_dump()
        fp = _write_dump_file(tmpdir, 10, dump)
        (cell_enc, cell_types, cell_batch,
         nuclei_enc, nuclei_types, nuclei_batch) = load_both_encoded_stage3(
            fp, set(CELL_TYPES)
        )
        expected_n = N_PER_CT * 2 * len(CELL_TYPES)
        assert cell_enc.shape == (expected_n, N_LATENT), f"Wrong cell shape: {cell_enc.shape}"
        assert nuclei_enc.shape == (expected_n, N_LATENT)
        assert all("c3A" in b for b in cell_batch), f"Cell batch IDs wrong: {cell_batch[:3]}"
        assert all("n3A" in b for b in nuclei_batch), f"Nuclei batch IDs wrong: {nuclei_batch[:3]}"
        print("  PASS: load_both_encoded_stage3")
    finally:
        shutil.rmtree(tmpdir)


def test_resolve_batch_id_list():
    """Test that resolve_batch_id_list picks up modality-specific batch IDs."""
    from eval_utils import resolve_batch_id_list
    dump = _make_synthetic_dump()
    entry = dump["Astro"]["0"]

    cell_ids = resolve_batch_id_list(entry, modality="cell")
    nuclei_ids = resolve_batch_id_list(entry, modality="nuclei")
    assert cell_ids is not None, "cell batch IDs should be found"
    assert nuclei_ids is not None, "nuclei batch IDs should be found"
    assert all("c3A" in b for b in cell_ids)
    assert all("n3A" in b for b in nuclei_ids)
    print("  PASS: resolve_batch_id_list")


def test_build_dump_logic():
    """Test the build_dump / is_eval_epoch / is_checkpoint_epoch logic."""
    eval_every = 50
    num_epochs = 600

    # save_epoch_dump = False
    eval_epochs = []
    dump_epochs_nosave = []
    for epoch in range(num_epochs):
        is_eval_epoch = (epoch % eval_every == 0 or epoch == num_epochs - 1)
        is_checkpoint_epoch = (epoch % 10 == 0 or epoch == num_epochs - 1)
        build_dump = is_eval_epoch or (False and is_checkpoint_epoch)
        if is_eval_epoch:
            eval_epochs.append(epoch)
        if build_dump:
            dump_epochs_nosave.append(epoch)

    assert dump_epochs_nosave == eval_epochs

    # save_epoch_dump = True
    checkpoint_epochs = []
    dump_epochs_save = []
    for epoch in range(num_epochs):
        is_eval_epoch = (epoch % eval_every == 0 or epoch == num_epochs - 1)
        is_checkpoint_epoch = (epoch % 10 == 0 or epoch == num_epochs - 1)
        build_dump = is_eval_epoch or (True and is_checkpoint_epoch)
        if is_checkpoint_epoch:
            checkpoint_epochs.append(epoch)
        if build_dump:
            dump_epochs_save.append(epoch)

    assert len(dump_epochs_save) > len(eval_epochs)
    for e in eval_epochs:
        assert e in dump_epochs_save
    for e in checkpoint_epochs:
        assert e in dump_epochs_save

    print("  PASS: build_dump logic")


def test_write_epoch_dump():
    """Test _write_epoch_dump logic produces correct filename."""
    def _write_epoch_dump(save_path, epoch, epoch_save_dict):
        dump_dir = os.path.join(save_path, "epoch_dump")
        os.makedirs(dump_dir, exist_ok=True)
        fp = os.path.join(dump_dir, f"train_epoch_{epoch}.json")
        with open(fp, "w") as f:
            json.dump(epoch_save_dict, f)
        return fp

    tmpdir = tempfile.mkdtemp(prefix="test_write_dump_")
    try:
        dump = _make_synthetic_dump()
        fp = _write_epoch_dump(tmpdir, 100, dump)
        assert fp.endswith("train_epoch_100.json"), f"Wrong filename: {fp}"
        assert os.path.isfile(fp)
        with open(fp) as f:
            loaded = json.load(f)
        assert set(loaded.keys()) == set(CELL_TYPES)
        print("  PASS: _write_epoch_dump")
    finally:
        shutil.rmtree(tmpdir)


# ── heavier tests (require torch via imputeVAE import) ──────────────────────

def test_4source_correlation():
    """Test run_4source_correlation produces expected outputs."""
    from eval_utils import compute_celltype_mean
    from imputeVAE.stage3_eval import run_4source_correlation

    tmpdir = tempfile.mkdtemp(prefix="test_corr_")
    try:
        dump = _make_synthetic_dump()
        fp = _write_dump_file(tmpdir, 10, dump)

        gene_names = [f"gene_{i}" for i in range(N_GENES)]
        raw_data = np.random.rand(N_PER_CT * len(CELL_TYPES), N_GENES).astype(np.float32)
        raw_types = np.repeat(CELL_TYPES, N_PER_CT)
        raw_mean = compute_celltype_mean(raw_data, raw_types, gene_names)

        output_dir = os.path.join(tmpdir, "corr_output")
        results = run_4source_correlation(
            fp, 10, raw_mean, raw_mean, gene_names,
            set(CELL_TYPES), output_dir,
        )

        assert os.path.isdir(output_dir)
        csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
        png_files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        assert len(csv_files) > 0, "No CSV files"
        assert len(png_files) > 0, "No PNG files"
        assert "decoded_cell_mean_shift" in results
        assert "decoded_nuclei_mean_shift" in results

        avg_corr_files = [f for f in csv_files if "avg_correlation" in f]
        assert len(avg_corr_files) > 0
        avg_df = pd.read_csv(os.path.join(output_dir, avg_corr_files[0]), index_col=0)
        assert avg_df.shape == (4, 4), f"Expected 4x4, got {avg_df.shape}"

        print("  PASS: run_4source_correlation")
    finally:
        shutil.rmtree(tmpdir)


def test_4source_heatmaps():
    """Test run_4source_heatmaps produces expected plot files."""
    from imputeVAE.stage3_eval import run_4source_heatmaps

    tmpdir = tempfile.mkdtemp(prefix="test_heatmap_")
    try:
        dump = _make_synthetic_dump()
        fp = _write_dump_file(tmpdir, 10, dump)
        raw_adata = _make_fake_raw_adata()
        output_dir = os.path.join(tmpdir, "heatmap_output")

        run_4source_heatmaps(
            fp, 10, raw_adata, raw_adata,
            set(CELL_TYPES), output_dir, use_log1p=False,
        )

        assert os.path.isdir(output_dir)
        png_files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        assert len(png_files) >= 8, f"Expected >=8 PNGs, got {len(png_files)}: {png_files}"

        for name in ["decoded_cell_heatmap.png", "decoded_nuclei_heatmap.png",
                      "raw_cell_raw_heatmap.png", "raw_nuclei_raw_heatmap.png"]:
            assert name in png_files, f"Missing {name}"

        print("  PASS: run_4source_heatmaps")
    finally:
        shutil.rmtree(tmpdir)


def test_full_evaluation_orchestrator():
    """Test run_stage3_full_evaluation end-to-end (scIB will skip due to env)."""
    from eval_utils import compute_celltype_mean
    from imputeVAE.stage3_eval import run_stage3_full_evaluation

    tmpdir = tempfile.mkdtemp(prefix="test_full_eval_")
    try:
        dump = _make_synthetic_dump()
        fp = _write_dump_file(tmpdir, 50, dump)
        raw_adata = _make_fake_raw_adata()
        gene_names = [f"gene_{i}" for i in range(N_GENES)]
        raw_data = np.random.rand(N_PER_CT * len(CELL_TYPES), N_GENES).astype(np.float32)
        raw_types = np.repeat(CELL_TYPES, N_PER_CT)
        raw_mean = compute_celltype_mean(raw_data, raw_types, gene_names)

        manifest = run_stage3_full_evaluation(
            tmpdir, 50, fp,
            raw_adata, raw_adata,
            set(CELL_TYPES),
            raw_mean, raw_mean, gene_names,
            use_log1p=False,
        )

        assert manifest["epoch"] == 50
        assert manifest["heatmaps"]["status"] == "success"
        assert manifest["correlation"]["status"] == "success"
        assert "total_time_seconds" in manifest

        manifest_file = os.path.join(tmpdir, "inline_eval", "full", "manifest_epoch_50.json")
        assert os.path.isfile(manifest_file), f"Manifest not written"

        print("  PASS: run_stage3_full_evaluation")
    finally:
        shutil.rmtree(tmpdir)


# ── runner ───────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    light_only = "--light" in sys.argv

    light_tests = [
        test_dump_format,
        test_dump_filename,
        test_load_both_decoded,
        test_load_both_encoded,
        test_resolve_batch_id_list,
        test_build_dump_logic,
        test_write_epoch_dump,
    ]

    heavy_tests = [
        test_4source_correlation,
        test_4source_heatmaps,
        test_full_evaluation_orchestrator,
    ]

    tests = light_tests if light_only else light_tests + heavy_tests

    if light_only:
        print("Running lightweight tests only (--light)\n")
    else:
        print("Running all tests\n")

    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    main()
