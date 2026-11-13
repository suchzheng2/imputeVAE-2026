"""
scButterfly baseline for cross-modality translation (nucleus -> cell RNA-seq).

Implements the scButterfly architecture (Cao et al. 2024, Nat Commun) adapted for
RNA->RNA translation. Uses the CITE-seq variant architecture: dual VAEs with
NetBlock encoders/decoders (no chromosome splitting) + translator + discriminator.

Architecture (from scButterfly source, train_model_cite.py):
  - Cell encoder: NetBlock [n_genes, 256, 128] + Single_Translator (128->128)
  - Nucleus encoder: NetBlock [n_genes, 256, 128] + Single_Translator (128->128)
  - Cell decoder: NetBlock [128, 256, n_genes]
  - Nucleus decoder: NetBlock [128, 256, n_genes]
  - Translator: dual-path VAE (128->128->128) mapping between latent spaces
  - Discriminators: NetBlock [128, 1] with Sigmoid

Training phases:
  Phase 1: Pretrain cell autoencoder (enc -> R_translator -> dec), MSE + KL
  Phase 2: Pretrain nucleus autoencoder (enc -> A_translator -> dec), MSE + KL
  Phase 3: Joint translation with adversarial loss

Pseudo-pairing: cell-type-matched random pairing (scButterfly-T augmentation).

Usage:
    python analysis/baseline_scbutterfly.py \
        --output_dir run/scbutterfly_baseline \
        --R2R_pretrain_epoch 100 --A2A_pretrain_epoch 100 \
        --translator_epoch 200

Requires: torch, anndata, scipy, numpy, pandas
"""

import os
import sys
import argparse
import json
import time
import random
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data paths (matches baseline_scvi.py)
# ---------------------------------------------------------------------------
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
CELL_DIR = "10X_cells_v2_v3_AIBS"
NUC_DIR = "10X_nuclei_v2_v3_AIBS"
DEFAULT_GENE_LIST = "data/training_gene_list_v3.csv"


# ---------------------------------------------------------------------------
# Architecture components (faithful reimplementation of scButterfly)
# ---------------------------------------------------------------------------

def setup_seed(seed=19193):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


class NetBlock(nn.Module):
    """MLP block with BN + activation + dropout. From scButterfly/model_component.py."""

    def __init__(self, nlayer, dim_list, act_list, dropout_rate, noise_rate):
        super().__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()

        for i in range(nlayer):
            self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            nn.init.xavier_uniform_(self.linear_list[i].weight)
            self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activation_list.append(act_list[i])
            if i != nlayer - 1:
                self.dropout_list.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.linear_list[i](x)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            if i != self.nlayer - 1:
                x = self.dropout_list[i](x)
        return x


class SingleTranslator(nn.Module):
    """Single-modality translator (VAE bottleneck) for pretraining."""

    def __init__(self, input_dim, embed_dim, act_list):
        super().__init__()
        mean_act, logvar_act, dec_act = act_list

        self.enc_mu = nn.Linear(input_dim, embed_dim)
        nn.init.xavier_uniform_(self.enc_mu.weight)
        self.bn_mu = nn.BatchNorm1d(embed_dim)
        self.act_mu = mean_act

        self.enc_d = nn.Linear(input_dim, embed_dim)
        nn.init.xavier_uniform_(self.enc_d.weight)
        self.bn_d = nn.BatchNorm1d(embed_dim)
        self.act_d = logvar_act

        self.dec = nn.Linear(embed_dim, input_dim)
        nn.init.xavier_uniform_(self.dec.weight)
        self.bn_dec = nn.BatchNorm1d(input_dim)
        self.act_dec = dec_act

    def forward(self, x, mode='train'):
        mu = self.act_mu(self.bn_mu(self.enc_mu(x)))
        d = self.act_d(self.bn_d(self.enc_d(x)))
        if mode == 'train':
            z = mu + torch.randn_like(mu) * torch.exp(d / 2)
        else:
            z = mu
        out = self.act_dec(self.bn_dec(self.dec(z)))
        return out, mu, d


class Translator(nn.Module):
    """Cross-modality translator VAE. From scButterfly/model_component.py."""

    def __init__(self, input_dim_r, input_dim_a, embed_dim, act_list):
        super().__init__()
        mean_act, logvar_act, dec_act = act_list

        # RNA path
        self.r_mu = nn.Linear(input_dim_r, embed_dim)
        nn.init.xavier_uniform_(self.r_mu.weight)
        self.r_bn_mu = nn.BatchNorm1d(embed_dim)
        self.r_act_mu = mean_act

        self.r_d = nn.Linear(input_dim_r, embed_dim)
        nn.init.xavier_uniform_(self.r_d.weight)
        self.r_bn_d = nn.BatchNorm1d(embed_dim)
        self.r_act_d = logvar_act

        self.r_dec = nn.Linear(embed_dim, input_dim_r)
        nn.init.xavier_uniform_(self.r_dec.weight)
        self.r_bn_dec = nn.BatchNorm1d(input_dim_r)
        self.r_act_dec = dec_act

        # ATAC (nucleus) path
        self.a_mu = nn.Linear(input_dim_a, embed_dim)
        nn.init.xavier_uniform_(self.a_mu.weight)
        self.a_bn_mu = nn.BatchNorm1d(embed_dim)
        self.a_act_mu = mean_act

        self.a_d = nn.Linear(input_dim_a, embed_dim)
        nn.init.xavier_uniform_(self.a_d.weight)
        self.a_bn_d = nn.BatchNorm1d(embed_dim)
        self.a_act_d = logvar_act

        self.a_dec = nn.Linear(embed_dim, input_dim_a)
        nn.init.xavier_uniform_(self.a_dec.weight)
        self.a_bn_dec = nn.BatchNorm1d(input_dim_a)
        self.a_act_dec = dec_act

    def forward(self, x, input_type, mode='train'):
        if input_type == 'RNA':
            mu = self.r_act_mu(self.r_bn_mu(self.r_mu(x)))
            d = self.r_act_d(self.r_bn_d(self.r_d(x)))
        else:  # ATAC / nucleus
            mu = self.a_act_mu(self.a_bn_mu(self.a_mu(x)))
            d = self.a_act_d(self.a_bn_d(self.a_d(x)))

        if mode == 'train':
            z = mu + torch.randn_like(mu) * torch.exp(d / 2)
        else:
            z = mu

        r_out = self.r_act_dec(self.r_bn_dec(self.r_dec(z)))
        a_out = self.a_act_dec(self.a_bn_dec(self.a_dec(z)))
        return r_out, a_out, mu, d


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PairedDataset(Dataset):
    """Dataset returning paired (cell, nucleus) samples by index."""

    def __init__(self, r_data, a_data, id_r, id_a):
        self.r_data = r_data
        self.a_data = a_data
        self.id_r = id_r
        self.id_a = id_a

    def __len__(self):
        return len(self.id_r)

    def __getitem__(self, idx):
        r = torch.from_numpy(self.r_data[self.id_r[idx]])
        a = torch.from_numpy(self.a_data[self.id_a[idx]])
        return torch.cat([r, a])


class SingleDataset(Dataset):
    """Dataset for single-modality data."""

    def __init__(self, data, id_list):
        self.data = data
        self.id_list = id_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[self.id_list[idx]])


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, loss):
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_filter(data_dir, batch_suffix, gene_list):
    """Load raw data, filter to batch + common genes + common CTs + gene list."""
    path = os.path.join(DATA_ROOT, data_dir, "data.h5ad")
    adata = ad.read_h5ad(path)

    if 'meta_batch_id' not in adata.obs:
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(
            lambda x: x.split('.')[0]
        )

    adata = adata[adata.obs['meta_batch_id'].str.endswith(batch_suffix)].copy()

    common_genes = pd.read_csv('data/common_genes.csv', header=0)['common_genes'].tolist()
    adata = adata[:, adata.var['gene_names'].isin(common_genes)].copy()

    common_ct = pd.read_csv('data/common_cell_types.csv', header=0)['common_cell_types'].tolist()
    adata = adata[adata.obs['cell_type'].isin(common_ct)].copy()

    adata = adata[:, adata.var['gene_names'].isin(gene_list)].copy()
    adata = adata[:, adata.var['gene_names'].argsort()].copy()

    print(f"  Loaded {data_dir} batch={batch_suffix}: {adata.shape}")
    return adata


def preprocess_rna(adata):
    """Apply scButterfly's default RNA preprocessing: normalize_total + log1p.

    No HVG selection — we use our Config B gene list instead.
    Returns AnnData with sparse X (required by scButterfly's Model).
    """
    if sparse.issparse(adata.X):
        X = adata.X.toarray().astype(np.float32)
    else:
        X = adata.X.astype(np.float32)

    # normalize_total (scanpy-style: divide by sum, multiply by 1e4)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X = X / row_sums * 1e4

    # log1p
    X = np.log1p(X)

    adata_p = ad.AnnData(
        X=sparse.csr_matrix(X),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    return adata_p


def create_pseudo_pairs(cell_cts, nuc_cts, seed=19193):
    """Create cell-type-matched pseudo-pairs (scButterfly-T augmentation style).

    For each cell type, randomly pair cells with nuclei. The smaller group is
    sampled with replacement to match the larger group, ensuring all data is used.

    Returns: (cell_indices, nuc_indices) of equal length.
    """
    rng = np.random.RandomState(seed)

    cell_by_ct = defaultdict(list)
    nuc_by_ct = defaultdict(list)

    for i, ct in enumerate(cell_cts):
        cell_by_ct[ct].append(i)
    for i, ct in enumerate(nuc_cts):
        nuc_by_ct[ct].append(i)

    common_cts = sorted(set(cell_by_ct.keys()) & set(nuc_by_ct.keys()))
    print(f"  Pseudo-pairing: {len(common_cts)} shared cell types")

    cell_ids = []
    nuc_ids = []

    for ct in common_cts:
        c = np.array(cell_by_ct[ct])
        n = np.array(nuc_by_ct[ct])
        max_n = max(len(c), len(n))

        # Sample with replacement from smaller group
        if len(c) < max_n:
            c_sampled = rng.choice(c, size=max_n, replace=True)
        else:
            c_sampled = c.copy()
            rng.shuffle(c_sampled)
        if len(n) < max_n:
            n_sampled = rng.choice(n, size=max_n, replace=True)
        else:
            n_sampled = n.copy()
            rng.shuffle(n_sampled)

        cell_ids.extend(c_sampled.tolist())
        nuc_ids.extend(n_sampled.tolist())

    print(f"  Total pseudo-pairs: {len(cell_ids)}")
    return cell_ids, nuc_ids


def augment_pairs(cell_ids, nuc_ids, cell_cts_arr, nuc_cts_arr, copy_count=3, seed=19193):
    """scButterfly-T style augmentation: shuffle within cell type, extend pairs.

    This matches scButterfly's cell_type_augmentation logic.
    """
    rng = random.Random(seed)

    # Group indices by cell type (using the paired indices)
    ct_to_cell = defaultdict(list)
    ct_to_nuc = defaultdict(list)
    for ci, ni in zip(cell_ids, nuc_ids):
        ct = cell_cts_arr[ci]
        ct_to_cell[ct].append(ci)
        ct_to_nuc[ct].append(ni)

    aug_cell = list(cell_ids)
    aug_nuc = list(nuc_ids)

    for ct in ct_to_cell:
        c_list = ct_to_cell[ct]
        n_list = ct_to_nuc[ct]
        for _ in range(copy_count - 1):
            c_copy = c_list.copy()
            n_copy = n_list.copy()
            rng.shuffle(c_copy)
            rng.shuffle(n_copy)
            aug_cell.extend(c_copy)
            aug_nuc.extend(n_copy)

    print(f"  After augmentation ({copy_count}x): {len(aug_cell)} pairs")
    return aug_cell, aug_nuc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_scbutterfly(
    cell_data, nuc_data,
    cell_cts, nuc_cts,
    n_genes,
    output_dir,
    R2R_pretrain_epoch=100,
    A2A_pretrain_epoch=100,
    translator_epoch=200,
    patience=50,
    batch_size=64,
    lr=0.001,
    discriminator_lr=0.005,
    dropout_rate=0.1,
    R_noise_rate=0.5,
    A_noise_rate=0.0,
    seed=19193,
):
    """Train scButterfly (CITE variant) for nucleus->cell translation."""
    setup_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Build model (matching CITE model defaults) ---
    # Encoder dims: [n_genes, 256, 128]
    # Decoder dims: [128, 256, n_genes]
    enc_dims_r = [n_genes, 256, 128]
    enc_dims_a = [n_genes, 256, 128]
    dec_dims_r = [128, 256, n_genes]
    dec_dims_a = [128, 256, n_genes]

    cell_encoder = NetBlock(2, enc_dims_r,
                            [nn.LeakyReLU(), nn.LeakyReLU()],
                            dropout_rate, R_noise_rate).to(device)
    nuc_encoder = NetBlock(2, enc_dims_a,
                           [nn.LeakyReLU(), nn.LeakyReLU()],
                           dropout_rate, A_noise_rate).to(device)
    cell_decoder = NetBlock(2, dec_dims_r,
                            [nn.LeakyReLU(), nn.LeakyReLU()],
                            dropout_rate, 0).to(device)
    # CITE model uses nn.Identity() for last ATAC decoder activation
    nuc_decoder = NetBlock(2, dec_dims_a,
                           [nn.LeakyReLU(), nn.Identity()],
                           dropout_rate, 0).to(device)

    translator_acts = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()]
    r_translator = SingleTranslator(128, 128, translator_acts).to(device)
    a_translator = SingleTranslator(128, 128, translator_acts).to(device)
    translator = Translator(128, 128, 128, translator_acts).to(device)

    disc_r = NetBlock(1, [128, 1], [nn.Sigmoid()], 0, 0).to(device)
    disc_a = NetBlock(1, [128, 1], [nn.Sigmoid()], 0, 0).to(device)

    # Count parameters
    all_modules = [cell_encoder, nuc_encoder, cell_decoder, nuc_decoder,
                   r_translator, a_translator, translator, disc_r, disc_a]
    total_params = sum(p.numel() for m in all_modules for p in m.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Create pseudo-pairs ---
    cell_ids, nuc_ids = create_pseudo_pairs(cell_cts, nuc_cts, seed=seed)
    cell_ids, nuc_ids = augment_pairs(
        cell_ids, nuc_ids, cell_cts, nuc_cts, copy_count=3, seed=seed
    )

    # Train/validation split (80/20, matching scButterfly default)
    n_total = len(cell_ids)
    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)
    n_train = int(n_total * 0.8)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_cell_ids = [cell_ids[i] for i in train_idx]
    train_nuc_ids = [nuc_ids[i] for i in train_idx]
    val_cell_ids = [cell_ids[i] for i in val_idx]
    val_nuc_ids = [nuc_ids[i] for i in val_idx]

    print(f"  Train pairs: {len(train_cell_ids)}, Val pairs: {len(val_cell_ids)}")

    # --- Loss setup ---
    r_loss_fn = nn.MSELoss()
    a_loss_fn = nn.MSELoss()
    d_loss_fn = nn.BCELoss()

    # KL weights (from scButterfly: 1/n_features * 20)
    R_kl_w = 1.0 / n_genes * 20
    A_kl_w = 1.0 / n_genes * 20
    kl_w = R_kl_w + A_kl_w
    # loss_weight = [r_loss, a_loss, d_loss, kl_R, kl_A, kl_all]
    loss_weight = [1.0, 1.0, 1.0, R_kl_w, A_kl_w, kl_w]

    t0 = time.time()

    # ===== Phase 1: Pretrain cell autoencoder =====
    print(f"\n--- Phase 1: Pretrain cell autoencoder ({R2R_pretrain_epoch} epochs) ---")
    cell_train_ds = SingleDataset(cell_data, train_cell_ids)
    cell_val_ds = SingleDataset(cell_data, val_cell_ids)
    cell_train_dl = DataLoader(cell_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    cell_val_dl = DataLoader(cell_val_ds, batch_size=batch_size, shuffle=False)

    opt_r = torch.optim.Adam(
        list(cell_encoder.parameters()) + list(cell_decoder.parameters()) +
        list(r_translator.parameters()), lr=lr
    )
    es_r = EarlyStopping(patience)

    for epoch in range(R2R_pretrain_epoch):
        cell_encoder.train(); cell_decoder.train(); r_translator.train()
        epoch_loss = 0
        kl_warmup = min(1.0, (epoch + 1) / 50)  # R_pretrain_kl_warmup=50

        for batch in cell_train_dl:
            batch = batch.to(device).float()
            latent = cell_encoder(batch)
            z, mu, d = r_translator(latent, 'train')
            recon = cell_decoder(z)
            recon_loss = r_loss_fn(recon, batch)
            kl = -0.5 * torch.mean(1 + d - mu.pow(2) - d.exp())
            loss = loss_weight[0] * recon_loss + loss_weight[3] * kl * kl_warmup
            opt_r.zero_grad()
            loss.backward()
            opt_r.step()
            epoch_loss += loss.item()

        # Validation
        cell_encoder.eval(); cell_decoder.eval(); r_translator.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in cell_val_dl:
                batch = batch.to(device).float()
                z, mu, d = r_translator(cell_encoder(batch), 'test')
                recon = cell_decoder(z)
                val_loss += r_loss_fn(recon, batch).item()
        val_loss /= max(len(cell_val_dl), 1)

        es_r.step(val_loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{R2R_pretrain_epoch}: train_loss={epoch_loss/len(cell_train_dl):.4f}, val_loss={val_loss:.4f}")
        if es_r.should_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # ===== Phase 2: Pretrain nucleus autoencoder =====
    print(f"\n--- Phase 2: Pretrain nucleus autoencoder ({A2A_pretrain_epoch} epochs) ---")
    nuc_train_ds = SingleDataset(nuc_data, train_nuc_ids)
    nuc_val_ds = SingleDataset(nuc_data, val_nuc_ids)
    nuc_train_dl = DataLoader(nuc_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    nuc_val_dl = DataLoader(nuc_val_ds, batch_size=batch_size, shuffle=False)

    opt_a = torch.optim.Adam(
        list(nuc_encoder.parameters()) + list(nuc_decoder.parameters()) +
        list(a_translator.parameters()), lr=lr
    )
    es_a = EarlyStopping(patience)

    for epoch in range(A2A_pretrain_epoch):
        nuc_encoder.train(); nuc_decoder.train(); a_translator.train()
        epoch_loss = 0
        kl_warmup = min(1.0, (epoch + 1) / 50)

        for batch in nuc_train_dl:
            batch = batch.to(device).float()
            latent = nuc_encoder(batch)
            z, mu, d = a_translator(latent, 'train')
            recon = nuc_decoder(z)
            recon_loss = a_loss_fn(recon, batch)
            kl = -0.5 * torch.mean(1 + d - mu.pow(2) - d.exp())
            loss = loss_weight[1] * recon_loss + loss_weight[4] * kl * kl_warmup
            opt_a.zero_grad()
            loss.backward()
            opt_a.step()
            epoch_loss += loss.item()

        nuc_encoder.eval(); nuc_decoder.eval(); a_translator.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in nuc_val_dl:
                batch = batch.to(device).float()
                z, mu, d = a_translator(nuc_encoder(batch), 'test')
                recon = nuc_decoder(z)
                val_loss += a_loss_fn(recon, batch).item()
        val_loss /= max(len(nuc_val_dl), 1)

        es_a.step(val_loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{A2A_pretrain_epoch}: train_loss={epoch_loss/len(nuc_train_dl):.4f}, val_loss={val_loss:.4f}")
        if es_a.should_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # ===== Phase 3: Joint translation training =====
    print(f"\n--- Phase 3: Translation training ({translator_epoch} epochs) ---")
    paired_train_ds = PairedDataset(cell_data, nuc_data, train_cell_ids, train_nuc_ids)
    paired_val_ds = PairedDataset(cell_data, nuc_data, val_cell_ids, val_nuc_ids)
    paired_train_dl = DataLoader(paired_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    paired_val_dl = DataLoader(paired_val_ds, batch_size=batch_size, shuffle=False)

    # Translator optimizer (all components)
    opt_trans = torch.optim.Adam(
        list(cell_encoder.parameters()) + list(nuc_encoder.parameters()) +
        list(cell_decoder.parameters()) + list(nuc_decoder.parameters()) +
        list(translator.parameters()), lr=lr
    )
    opt_disc = torch.optim.Adam(
        list(disc_r.parameters()) + list(disc_a.parameters()), lr=discriminator_lr
    )
    es_t = EarlyStopping(patience)

    for epoch in range(translator_epoch):
        for m in all_modules:
            m.train()
        epoch_loss = 0
        kl_warmup = min(1.0, (epoch + 1) / 50)

        for batch in paired_train_dl:
            batch = batch.to(device).float()
            r_input = batch[:, :n_genes]
            a_input = batch[:, n_genes:]

            # Forward through encoders
            r_enc = cell_encoder(r_input)
            a_enc = nuc_encoder(a_input)

            # Forward through translator
            r2r, r2a, mu_r, d_r = translator(r_enc, 'RNA', 'train')
            a2r, a2a, mu_a, d_a = translator(a_enc, 'ATAC', 'train')

            # Decode
            r2r_dec = cell_decoder(r2r)
            r2a_dec = nuc_decoder(r2a)
            a2r_dec = cell_decoder(a2r)
            a2a_dec = nuc_decoder(a2a)

            # Reconstruction losses
            r_recon = r_loss_fn(r2r_dec, r_input)
            a_recon = a_loss_fn(a2a_dec, a_input)

            # Cross-modal reconstruction losses
            r2a_loss = a_loss_fn(r2a_dec, a_input)
            a2r_loss = r_loss_fn(a2r_dec, r_input)

            # KL divergence
            kl_r = -0.5 * torch.mean(1 + d_r - mu_r.pow(2) - d_r.exp())
            kl_a = -0.5 * torch.mean(1 + d_a - mu_a.pow(2) - d_a.exp())

            # Discriminator loss (adversarial)
            real_r = disc_r(r2r.detach())
            fake_r = disc_r(a2r.detach())
            real_a = disc_a(a2a.detach())
            fake_a = disc_a(r2a.detach())

            d_loss_r = d_loss_fn(real_r, torch.ones_like(real_r)) + \
                       d_loss_fn(fake_r, torch.zeros_like(fake_r))
            d_loss_a = d_loss_fn(real_a, torch.ones_like(real_a)) + \
                       d_loss_fn(fake_a, torch.zeros_like(fake_a))

            # Update discriminator
            opt_disc.zero_grad()
            (d_loss_r + d_loss_a).backward()
            opt_disc.step()

            # Generator (fool discriminator) loss
            fake_r_g = disc_r(a2r)
            fake_a_g = disc_a(r2a)
            g_loss = d_loss_fn(fake_r_g, torch.ones_like(fake_r_g)) + \
                     d_loss_fn(fake_a_g, torch.ones_like(fake_a_g))

            # Total translator loss
            total = (loss_weight[0] * (r_recon + a2r_loss) +
                     loss_weight[1] * (a_recon + r2a_loss) +
                     loss_weight[2] * g_loss +
                     loss_weight[5] * (kl_r + kl_a) * kl_warmup)

            opt_trans.zero_grad()
            total.backward()
            opt_trans.step()
            epoch_loss += total.item()

        # Validation
        for m in all_modules:
            m.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in paired_val_dl:
                batch = batch.to(device).float()
                r_input = batch[:, :n_genes]
                a_input = batch[:, n_genes:]
                r_enc = cell_encoder(r_input)
                a_enc = nuc_encoder(a_input)
                r2r, r2a, _, _ = translator(r_enc, 'RNA', 'test')
                a2r, a2a, _, _ = translator(a_enc, 'ATAC', 'test')
                r2r_dec = cell_decoder(r2r)
                a2r_dec = cell_decoder(a2r)
                a2a_dec = nuc_decoder(a2a)
                val_loss += (r_loss_fn(r2r_dec, r_input).item() +
                             a_loss_fn(a2a_dec, a_input).item() +
                             r_loss_fn(a2r_dec, r_input).item())
        val_loss /= max(len(paired_val_dl), 1)

        es_t.step(val_loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{translator_epoch}: train_loss={epoch_loss/len(paired_train_dl):.4f}, val_loss={val_loss:.4f}")
        if es_t.should_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # --- Save model ---
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(cell_encoder.state_dict(), os.path.join(model_dir, "cell_encoder.pt"))
    torch.save(nuc_encoder.state_dict(), os.path.join(model_dir, "nuc_encoder.pt"))
    torch.save(cell_decoder.state_dict(), os.path.join(model_dir, "cell_decoder.pt"))
    torch.save(nuc_decoder.state_dict(), os.path.join(model_dir, "nuc_decoder.pt"))
    torch.save(translator.state_dict(), os.path.join(model_dir, "translator.pt"))

    return (cell_encoder, nuc_encoder, cell_decoder, nuc_decoder,
            translator, total_params, elapsed)


def predict_nucleus_to_cell(nuc_data, nuc_encoder, translator, cell_decoder,
                            batch_size=256):
    """Translate all nuclei to cell expression.

    Path: nucleus -> nuc_encoder -> translator(ATAC->RNA) -> cell_decoder -> predicted cell
    """
    device = next(nuc_encoder.parameters()).device
    nuc_encoder.eval()
    translator.eval()
    cell_decoder.eval()

    all_preds = []
    n = nuc_data.shape[0]
    ds = SingleDataset(nuc_data, list(range(n)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dl, desc="Predicting nucleus->cell"):
            batch = batch.to(device).float()
            enc = nuc_encoder(batch)
            r_out, _, _, _ = translator(enc, 'ATAC', 'test')
            pred = cell_decoder(r_out)
            all_preds.append(pred.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="scButterfly baseline for nucleus->cell translation"
    )
    parser.add_argument("--output_dir", type=str, default="run/scbutterfly_baseline")
    parser.add_argument("--gene_list", type=str, default=DEFAULT_GENE_LIST)
    parser.add_argument("--R2R_pretrain_epoch", type=int, default=100)
    parser.add_argument("--A2A_pretrain_epoch", type=int, default=100)
    parser.add_argument("--translator_epoch", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=19193)
    args = parser.parse_args()

    print("=" * 60)
    print("scButterfly baseline (CITE variant, RNA->RNA)")
    print("=" * 60)

    # Load data
    gene_list = pd.read_csv(args.gene_list, header=None)[0].tolist()
    print(f"\nLoading data ({len(gene_list)} genes)...")
    adata_cell = load_and_filter(CELL_DIR, "c3A", gene_list)
    adata_nuc = load_and_filter(NUC_DIR, "n3A", gene_list)

    n_genes = adata_cell.shape[1]

    # Preprocess (normalize_total + log1p, matching scButterfly default)
    print("Preprocessing...")
    adata_cell_p = preprocess_rna(adata_cell)
    adata_nuc_p = preprocess_rna(adata_nuc)

    # Extract numpy arrays
    cell_data = adata_cell_p.X.toarray().astype(np.float32)
    nuc_data = adata_nuc_p.X.toarray().astype(np.float32)
    cell_cts = adata_cell_p.obs['cell_type'].values
    nuc_cts = adata_nuc_p.obs['cell_type'].values

    print(f"\nCell data: {cell_data.shape}")
    print(f"Nucleus data: {nuc_data.shape}")
    print(f"Cell types (cell): {len(set(cell_cts))}")
    print(f"Cell types (nucleus): {len(set(nuc_cts))}")

    # Train
    (cell_encoder, nuc_encoder, cell_decoder, nuc_decoder,
     translator_model, total_params, elapsed) = train_scbutterfly(
        cell_data, nuc_data, cell_cts, nuc_cts, n_genes,
        output_dir=args.output_dir,
        R2R_pretrain_epoch=args.R2R_pretrain_epoch,
        A2A_pretrain_epoch=args.A2A_pretrain_epoch,
        translator_epoch=args.translator_epoch,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    # Predict
    print("\nGenerating nucleus->cell translations...")
    imputed = predict_nucleus_to_cell(
        nuc_data, nuc_encoder, translator_model, cell_decoder,
        batch_size=256
    )
    print(f"Imputed matrix shape: {imputed.shape}")

    # Save outputs (same format as other baselines)
    np.save(os.path.join(args.output_dir, "imputed_cell_from_nucleus.npy"),
            imputed.astype(np.float32))

    nuc_obs = adata_nuc_p.obs[["cell_type"]].copy()
    nuc_obs["modality"] = "nucleus"
    nuc_obs.to_csv(os.path.join(args.output_dir, "nucleus_obs.csv"))

    cell_obs = adata_cell_p.obs[["cell_type"]].copy()
    cell_obs["modality"] = "cell"
    cell_obs.to_csv(os.path.join(args.output_dir, "cell_obs.csv"))

    gene_names = adata_cell_p.var['gene_names'].values
    pd.Series(gene_names).to_csv(
        os.path.join(args.output_dir, "gene_names.csv"),
        index=False, header=False
    )

    # Save args
    args_dict = {
        "method": "scbutterfly",
        "variant": "CITE (RNA-RNA, NetBlock both sides)",
        "architecture": "dual-VAE + translator + discriminator",
        "n_genes": n_genes,
        "n_cells": int(cell_data.shape[0]),
        "n_nuclei": int(nuc_data.shape[0]),
        "total_params": total_params,
        "training_time_s": elapsed,
        "R2R_pretrain_epoch": args.R2R_pretrain_epoch,
        "A2A_pretrain_epoch": args.A2A_pretrain_epoch,
        "translator_epoch": args.translator_epoch,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "encoder_dims": "[n_genes, 256, 128]",
        "decoder_dims": "[128, 256, n_genes]",
        "translator_embed_dim": 128,
        "loss": "MSE (both sides) + BCE (discriminator) + KL",
        "augmentation": "cell_type (3x, scButterfly-T)",
        "preprocessing": "normalize_total(1e4) + log1p (no HVG, Config B genes)",
    }
    with open(os.path.join(args.output_dir, "arguments.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    print(f"\nOutputs saved to {args.output_dir}/")
    print("Run evaluation with:")
    print(f"  python analysis/evaluate_baseline_incremental.py \\")
    print(f"    --imputed_npy {args.output_dir}/imputed_cell_from_nucleus.npy \\")
    print(f"    --obs_csv {args.output_dir}/nucleus_obs.csv \\")
    print(f"    --name scbutterfly --output_dir {args.output_dir}/eval")


if __name__ == "__main__":
    main()
