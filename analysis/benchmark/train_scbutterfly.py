"""
scButterfly baseline (split-aware).  Dual VAE + translator + discriminators with
cell-type pseudo-pairing, trained on the TRAINING partition only.

Change from the old benchmark: inputs/outputs are log1p(raw counts), the same
space as the cVAE and as the evaluation reference.  The old run used
normalize_total(1e4)+log1p, so its output lived on a different absolute scale and
every scale-sensitive metric (R2, MSE, Wasserstein) was penalised for a
preprocessing choice rather than for model quality.  The MSE objective does not
depend on which log-scale is used.

Usage:
    python -m analysis.benchmark.train_scbutterfly --seed 19193 --eval_partition test \
        --output_dir run/benchmark/test/scbutterfly_seed0
"""

import os
import sys
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.benchmark.data_split import load_benchmark_data, save_standard_outputs
from analysis.baseline_scbutterfly import (
    NetBlock, SingleTranslator, Translator, PairedDataset, SingleDataset,
    EarlyStopping, create_pseudo_pairs, augment_pairs, setup_seed,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R2R_pretrain_epoch", type=int, default=100)
    p.add_argument("--A2A_pretrain_epoch", type=int, default=100)
    p.add_argument("--translator_epoch", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=19193)
    p.add_argument("--eval_partition", default="test", choices=["valid", "test"])
    p.add_argument("--output_dir", default="run/benchmark/test/scbutterfly_seed0")
    args = p.parse_args()

    print("=" * 60)
    print(f"Split-aware benchmark: scButterfly (seed {args.seed})")
    print("=" * 60)
    data = load_benchmark_data(args.eval_partition, return_raw_counts=False, include_eval_cells=False)
    gene_names = data["gene_names"]
    n_genes = len(gene_names)

    cell_data = np.ascontiguousarray(data["cell_train"].X, dtype=np.float32)
    nuc_train_data = np.ascontiguousarray(data["nuc_train"].X, dtype=np.float32)
    nuc_eval_data = np.ascontiguousarray(data["nuc_eval"].X, dtype=np.float32)
    cell_cts = data["cell_train"].obs["cell_type"].values.astype(str)
    nuc_train_cts = data["nuc_train"].obs["cell_type"].values.astype(str)
    print(f"Train: {cell_data.shape[0]} cells, {nuc_train_data.shape[0]} nuclei; eval: {nuc_eval_data.shape[0]} nuclei")

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout_rate, R_noise_rate, A_noise_rate = 0.1, 0.5, 0.0
    cell_encoder = NetBlock(2, [n_genes, 256, 128], [nn.LeakyReLU(), nn.LeakyReLU()], dropout_rate, R_noise_rate).to(device)
    nuc_encoder = NetBlock(2, [n_genes, 256, 128], [nn.LeakyReLU(), nn.LeakyReLU()], dropout_rate, A_noise_rate).to(device)
    cell_decoder = NetBlock(2, [128, 256, n_genes], [nn.LeakyReLU(), nn.LeakyReLU()], dropout_rate, 0).to(device)
    nuc_decoder = NetBlock(2, [128, 256, n_genes], [nn.LeakyReLU(), nn.Identity()], dropout_rate, 0).to(device)
    acts = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()]
    r_translator = SingleTranslator(128, 128, acts).to(device)
    a_translator = SingleTranslator(128, 128, acts).to(device)
    translator = Translator(128, 128, 128, acts).to(device)
    disc_r = NetBlock(1, [128, 1], [nn.Sigmoid()], 0, 0).to(device)
    disc_a = NetBlock(1, [128, 1], [nn.Sigmoid()], 0, 0).to(device)
    all_modules = [cell_encoder, nuc_encoder, cell_decoder, nuc_decoder, r_translator, a_translator,
                   translator, disc_r, disc_a]
    total_params = sum(p.numel() for m in all_modules for p in m.parameters())
    print(f"Total parameters: {total_params:,}")

    cell_ids, nuc_ids = create_pseudo_pairs(cell_cts, nuc_train_cts, seed=args.seed)
    cell_ids, nuc_ids = augment_pairs(cell_ids, nuc_ids, cell_cts, nuc_train_cts, copy_count=3, seed=args.seed)
    n_total = len(cell_ids)
    indices = list(range(n_total))
    random.Random(args.seed).shuffle(indices)
    n_train = int(n_total * 0.8)
    train_cell_ids = [cell_ids[indices[i]] for i in range(n_train)]
    train_nuc_ids = [nuc_ids[indices[i]] for i in range(n_train)]
    val_cell_ids = [cell_ids[indices[i]] for i in range(n_train, n_total)]
    val_nuc_ids = [nuc_ids[indices[i]] for i in range(n_train, n_total)]

    r_loss_fn, a_loss_fn, d_loss_fn = nn.MSELoss(), nn.MSELoss(), nn.BCELoss()
    R_kl_w = A_kl_w = 1.0 / n_genes * 20
    loss_weight = [1.0, 1.0, 1.0, R_kl_w, A_kl_w, R_kl_w + A_kl_w]
    t0 = time.time()

    def pretrain(name, enc, dec, trans, Xdata, tr_ids, va_ids, n_epochs, loss_fn, klw):
        print(f"\n--- {name} autoencoder pretraining ({n_epochs} epochs max) ---")
        tr_dl = DataLoader(SingleDataset(Xdata, tr_ids), batch_size=args.batch_size, shuffle=True, drop_last=True)
        va_dl = DataLoader(SingleDataset(Xdata, va_ids), batch_size=args.batch_size, shuffle=False)
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(trans.parameters()), lr=args.lr)
        es = EarlyStopping(args.patience)
        for epoch in range(n_epochs):
            enc.train(); dec.train(); trans.train()
            tot = 0.0
            kl_warmup = min(1.0, (epoch + 1) / 50)
            for batch in tr_dl:
                batch = batch.to(device).float()
                z, mu, d = trans(enc(batch), "train")
                loss = loss_fn(dec(z), batch) + klw * (-0.5 * torch.mean(1 + d - mu.pow(2) - d.exp())) * kl_warmup
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            enc.eval(); dec.eval(); trans.eval()
            vl = 0.0
            with torch.no_grad():
                for batch in va_dl:
                    batch = batch.to(device).float()
                    z, mu, d = trans(enc(batch), "test")
                    vl += loss_fn(dec(z), batch).item()
            vl /= max(len(va_dl), 1)
            es.step(vl)
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: train={tot/len(tr_dl):.4f} val={vl:.4f}", flush=True)
            if es.should_stop:
                print(f"  Early stopping at epoch {epoch+1}"); break

    pretrain("Cell", cell_encoder, cell_decoder, r_translator, cell_data, train_cell_ids, val_cell_ids,
             args.R2R_pretrain_epoch, r_loss_fn, loss_weight[3])
    pretrain("Nucleus", nuc_encoder, nuc_decoder, a_translator, nuc_train_data, train_nuc_ids, val_nuc_ids,
             args.A2A_pretrain_epoch, a_loss_fn, loss_weight[4])

    print(f"\n--- Translation ({args.translator_epoch} epochs max) ---")
    tr_dl = DataLoader(PairedDataset(cell_data, nuc_train_data, train_cell_ids, train_nuc_ids),
                       batch_size=args.batch_size, shuffle=True, drop_last=True)
    va_dl = DataLoader(PairedDataset(cell_data, nuc_train_data, val_cell_ids, val_nuc_ids),
                       batch_size=args.batch_size, shuffle=False)
    opt_t = torch.optim.Adam(list(cell_encoder.parameters()) + list(nuc_encoder.parameters()) +
                             list(cell_decoder.parameters()) + list(nuc_decoder.parameters()) +
                             list(translator.parameters()), lr=args.lr)
    opt_d = torch.optim.Adam(list(disc_r.parameters()) + list(disc_a.parameters()), lr=0.005)
    es = EarlyStopping(args.patience)
    epochs_run = 0
    for epoch in range(args.translator_epoch):
        for m in all_modules: m.train()
        tot = 0.0
        kl_warmup = min(1.0, (epoch + 1) / 50)
        for batch in tr_dl:
            batch = batch.to(device).float()
            r_in, a_in = batch[:, :n_genes], batch[:, n_genes:]
            r_enc, a_enc = cell_encoder(r_in), nuc_encoder(a_in)
            r2r, r2a, mu_r, d_r = translator(r_enc, "RNA", "train")
            a2r, a2a, mu_a, d_a = translator(a_enc, "ATAC", "train")
            ones = torch.ones(r2r.size(0), 1, device=device); zeros = torch.zeros(r2r.size(0), 1, device=device)
            d_loss = (d_loss_fn(disc_r(r2r.detach()), ones) + d_loss_fn(disc_r(a2r.detach()), zeros) +
                      d_loss_fn(disc_a(a2a.detach()), ones) + d_loss_fn(disc_a(r2a.detach()), zeros))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            g_loss = d_loss_fn(disc_r(a2r), ones) + d_loss_fn(disc_a(r2a), ones)
            kl = (-0.5 * torch.mean(1 + d_r - mu_r.pow(2) - d_r.exp())) + (-0.5 * torch.mean(1 + d_a - mu_a.pow(2) - d_a.exp()))
            total = (loss_weight[0] * (r_loss_fn(cell_decoder(r2r), r_in) + r_loss_fn(cell_decoder(a2r), r_in)) +
                     loss_weight[1] * (a_loss_fn(nuc_decoder(a2a), a_in) + a_loss_fn(nuc_decoder(r2a), a_in)) +
                     loss_weight[2] * g_loss + loss_weight[5] * kl * kl_warmup)
            opt_t.zero_grad(); total.backward(); opt_t.step()
            tot += total.item()
        for m in all_modules: m.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in va_dl:
                batch = batch.to(device).float()
                r_in, a_in = batch[:, :n_genes], batch[:, n_genes:]
                r2r, _, _, _ = translator(cell_encoder(r_in), "RNA", "test")
                a2r, a2a, _, _ = translator(nuc_encoder(a_in), "ATAC", "test")
                vl += (r_loss_fn(cell_decoder(r2r), r_in).item() + a_loss_fn(nuc_decoder(a2a), a_in).item() +
                       r_loss_fn(cell_decoder(a2r), r_in).item())
        vl /= max(len(va_dl), 1)
        es.step(vl)
        epochs_run = epoch + 1
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: train={tot/len(tr_dl):.4f} val={vl:.4f}", flush=True)
        if es.should_stop:
            print(f"  Early stopping at epoch {epoch+1}"); break
    elapsed = time.time() - t0
    print(f"\nTraining: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    model_dir = os.path.join(args.output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    for name, m in [("cell_encoder", cell_encoder), ("nuc_encoder", nuc_encoder), ("cell_decoder", cell_decoder),
                    ("nuc_decoder", nuc_decoder), ("translator", translator)]:
        torch.save(m.state_dict(), os.path.join(model_dir, f"{name}.pt"))

    nuc_encoder.eval(); translator.eval(); cell_decoder.eval()
    preds = []
    dl = DataLoader(SingleDataset(nuc_eval_data, list(range(nuc_eval_data.shape[0]))), batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(dl, desc="Predicting"):
            r_out, _, _, _ = translator(nuc_encoder(batch.to(device).float()), "ATAC", "test")
            preds.append(cell_decoder(r_out).cpu().numpy())
    imputed = np.clip(np.concatenate(preds, axis=0), 0, None).astype(np.float32)

    args_dict = {
        "method": "scButterfly", "variant": "CITE-style RNA-RNA NetBlock", "seed": args.seed,
        "n_genes": n_genes, "n_params": int(total_params), "training_time_s": elapsed,
        "R2R_pretrain_epoch": args.R2R_pretrain_epoch, "A2A_pretrain_epoch": args.A2A_pretrain_epoch,
        "translator_epoch": args.translator_epoch, "translator_epochs_run": epochs_run,
        "patience": args.patience, "batch_size": args.batch_size, "lr": args.lr,
        "preprocessing": "log1p(raw counts) (same space as cVAE and reference)",
        "checkpoint": "last epoch (early stopping on 20% of training pseudo-pairs)",
        "eval_partition": args.eval_partition,
    }
    save_standard_outputs(args.output_dir, imputed, data["nuc_eval"].obs[["obs_id", "cell_type", "donor"]],
                          gene_names, args_dict, data["split_info"])
    print("Done.")


if __name__ == "__main__":
    main()
