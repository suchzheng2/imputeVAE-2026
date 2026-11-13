import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import umap
import re
import os

def plot_loss_after_train_general(loss_dict, save_path=None):
    """
    Plot loss after training
    
    Args:
        loss_data: Loss data
        save_path: Path to save the plot
    """
    os.makedirs(save_path, exist_ok=True)
    for evalutation, loss_data in loss_dict.items():
        epoch_list = range(len(loss_data))
        # plot curve and save
        plt.plot(epoch_list, loss_data, label=evalutation)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{evalutation} Loss Curve')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'{evalutation}_loss_curve.png'))
        plt.close()

def plot_loss_curve_stage3_v2(save_path):
    """Plot stage-3 loss curves from the aggregated epoch-loss log.

    Generates three sets of plots:
    1. Per-loss-term train-only curves
    2. Per-loss-term train-vs-valid overlay curves (if valid data exists)
    3. All-terms-combined plots (train-only and train-vs-valid)
    """
    loss_file = os.path.join(save_path, "train_epoch_loss", "loss.txt")
    if not os.path.exists(loss_file):
        print(f"Stage 3 loss file not found: {loss_file}")
        return

    train_df = pd.read_csv(loss_file)
    if "epoch" not in train_df.columns:
        print(f"Stage 3 loss file is missing the epoch column: {loss_file}")
        return

    loss_curves_dir = os.path.join(save_path, "loss_curves")
    os.makedirs(loss_curves_dir, exist_ok=True)

    loss_columns = [
        "loss",
        "nuclei_recons_loss",
        "nuclei_JSD_div",
        "cell_recons_loss",
        "cell_JSD_div",
        "alignment_JSD_loss",
        "cross_recons_loss",
    ]
    available_columns = [col for col in loss_columns if col in train_df.columns]

    if not available_columns:
        print(f"No stage 3 loss columns found in: {loss_file}")
        return

    # Load valid loss if available
    valid_file = os.path.join(save_path, "valid_epoch_loss", "loss.txt")
    valid_df = None
    if os.path.exists(valid_file):
        try:
            valid_df = pd.read_csv(valid_file)
            if "epoch" not in valid_df.columns:
                valid_df = None
        except Exception:
            valid_df = None

    # 1. Per-column train-only plots
    for column in available_columns:
        plt.figure(figsize=(8, 5))
        plt.plot(train_df["epoch"], train_df[column], linewidth=2, label=column)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{column} Over Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(loss_curves_dir, f"{column}_loss_curve.png"))
        plt.close()

    # 2. Per-column train-vs-valid overlay plots
    if valid_df is not None:
        valid_available = [col for col in available_columns if col in valid_df.columns]
        for column in valid_available:
            plt.figure(figsize=(8, 5))
            plt.plot(train_df["epoch"], train_df[column], linewidth=2, label=f"train {column}")
            plt.plot(valid_df["epoch"], valid_df[column], linewidth=2, linestyle="--",
                     label=f"valid {column}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{column} — Train vs Valid")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(loss_curves_dir, f"{column}_train_vs_valid.png"))
            plt.close()

    # 3. All-terms combined (train only)
    plt.figure(figsize=(10, 6))
    for column in available_columns:
        plt.plot(train_df["epoch"], train_df[column], linewidth=2, label=column)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage 3 Loss Terms Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(loss_curves_dir, "stage3_all_losses.png"))
    plt.close()

    # 4. All-terms combined (train vs valid)
    if valid_df is not None:
        valid_available = [col for col in available_columns if col in valid_df.columns]
        if valid_available:
            plt.figure(figsize=(12, 7))
            colors = plt.cm.tab10(np.linspace(0, 1, len(valid_available)))
            for i, column in enumerate(valid_available):
                plt.plot(train_df["epoch"], train_df[column], linewidth=2,
                         color=colors[i], label=f"train {column}")
                plt.plot(valid_df["epoch"], valid_df[column], linewidth=2,
                         color=colors[i], linestyle="--", label=f"valid {column}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Stage 3 Loss Terms — Train vs Valid")
            plt.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(loss_curves_dir, "stage3_all_losses_train_vs_valid.png"))
            plt.close()


def plot_loss_after_train(save_path):
    """
    Plot loss after training
    
    Args:
        save_path: Path to save the plot
    """
    pass

    # Load loss data
    loss_data = load_batch_losses(save_path)

    # Plot loss curve
    plt.plot(loss_data['epoch'], loss_data['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))

def process_batch_ids(adata):
    """
    Process batch_id column: remove dot and number suffix, extract meta_batch_id
    
    Args:
        adata: AnnData object with 'batch_id' column in obs
    
    Returns:
        Modified AnnData object with processed columns
    """
    # Copy to avoid modifying original
    adata = adata.copy()
    
    # Remove dot and number at the end (e.g., "1-n3c.1" -> "1-n3c")
    adata.obs['batch_id_clean'] = adata.obs['batch_id'].str.replace(r'\.\d+$', '', regex=True)
    
    # Extract last 3 characters as meta_batch_id (e.g., "1-n3c" -> "n3c")
    adata.obs['meta_batch_id'] = adata.obs['batch_id_clean'].str[-3:]
    
    return adata


def select_highly_variable_genes(adata, n_top_genes=3000):
    """
    Select highly variable genes to reduce memory usage
    
    Args:
        adata: AnnData object
        n_top_genes: Number of top variable genes to select (default: 3000)
    
    Returns:
        AnnData object with selected genes
    """
    print(f"Selecting highly variable genes...")
    print(f"Original number of genes: {adata.shape[1]}")
    
    # Calculate variance for each gene
    if sparse.issparse(adata.X):
        # For sparse matrices
        mean = np.array(adata.X.mean(axis=0)).flatten()
        mean_sq = np.array(adata.X.power(2).mean(axis=0)).flatten()
        gene_vars = mean_sq - mean ** 2
    else:
        gene_vars = np.var(adata.X, axis=0)
    
    # Select top variable genes
    top_gene_indices = np.argsort(gene_vars)[-n_top_genes:]
    adata_hvg = adata[:, top_gene_indices].copy()
    
    print(f"Selected {n_top_genes} highly variable genes")
    return adata_hvg


def combine_cell_nuclei_data(cells_adata, nuclei_adata, use_log1p=True, n_top_genes=None):
    """
    Combine cell and nuclei datasets for joint analysis
    
    Args:
        cells_adata: AnnData object for cells
        nuclei_adata: AnnData object for nuclei
        use_log1p: Whether to use log1p layer (default: True)
        n_top_genes: Number of highly variable genes to select (optional)
    
    Returns:
        Combined AnnData object with modality annotation
    """
    # Process batch IDs for both datasets
    cells_adata = process_batch_ids(cells_adata)
    nuclei_adata = process_batch_ids(nuclei_adata)
    
    # Add modality label
    cells_adata.obs['modality'] = 'cell'
    nuclei_adata.obs['modality'] = 'nuclei'
    
    # Find common genes (variables)
    common_genes = cells_adata.var_names.intersection(nuclei_adata.var_names)
    print(f"Common genes: {len(common_genes)} out of {len(cells_adata.var_names)} (cells) and {len(nuclei_adata.var_names)} (nuclei)")
    
    # Subset to common genes
    cells_subset = cells_adata[:, common_genes].copy()
    nuclei_subset = nuclei_adata[:, common_genes].copy()
    
    # Use log1p layer if specified, otherwise use X
    if use_log1p and 'log1p' in cells_adata.layers:
        cells_subset.X = cells_subset.layers['log1p']
        nuclei_subset.X = nuclei_subset.layers['log1p']
    
    # Concatenate datasets
    combined_adata = ad.concat(
        [cells_subset, nuclei_subset],
        join='outer',
        label='source',
        keys=['cells', 'nuclei'],
        index_unique='_'
    )
    
    print(f"\nCombined dataset shape: {combined_adata.shape}")
    print(f"Modality counts:\n{combined_adata.obs['modality'].value_counts()}")
    print(f"Meta batch ID counts:\n{combined_adata.obs['meta_batch_id'].value_counts()}")
    print(f"Cleaned batch ID (unique): {combined_adata.obs['batch_id_clean'].nunique()}")
    
    # Select highly variable genes if specified
    if n_top_genes is not None:
        combined_adata = select_highly_variable_genes(combined_adata, n_top_genes=n_top_genes)
    
    return combined_adata


def compute_umap(adata, n_pcs=50, n_neighbors=15, min_dist=0.3, random_state=42, 
                 max_cells=None, use_sparse=True):
    """
    Compute UMAP embedding with memory-efficient options
    
    Args:
        adata: AnnData object
        n_pcs: Number of principal components (default: 50)
        n_neighbors: Number of neighbors for UMAP (default: 15)
        min_dist: Minimum distance for UMAP (default: 0.3)
        random_state: Random state for reproducibility (default: 42)
        max_cells: Maximum number of cells to use (downsamples if exceeded)
        use_sparse: Use sparse-compatible methods (default: True)
    
    Returns:
        AnnData object with UMAP coordinates in obsm['X_umap']
    """
    print(f"Original dataset size: {adata.shape}")
    
    # Downsample if dataset is too large
    if max_cells is not None and adata.shape[0] > max_cells:
        print(f"Downsampling to {max_cells} cells for visualization...")
        
        # Stratified sampling to maintain proportions across cell types and modalities
        if 'cell_type' in adata.obs.columns and 'modality' in adata.obs.columns:
            indices = []
            for ct in adata.obs['cell_type'].unique():
                for mod in adata.obs['modality'].unique():
                    mask = (adata.obs['cell_type'] == ct) & (adata.obs['modality'] == mod)
                    ct_mod_indices = np.where(mask)[0]
                    if len(ct_mod_indices) > 0:
                        n_sample = int(len(ct_mod_indices) * (max_cells / adata.shape[0]))
                        n_sample = max(1, n_sample)  # At least 1 cell per group
                        sampled = np.random.choice(ct_mod_indices, 
                                                   size=min(n_sample, len(ct_mod_indices)), 
                                                   replace=False)
                        indices.extend(sampled)
            adata = adata[indices, :].copy()
        else:
            indices = np.random.choice(adata.shape[0], size=max_cells, replace=False)
            adata = adata[indices, :].copy()
        
        print(f"Downsampled to: {adata.shape}")
    
    print("Computing PCA...")
    
    # Choose method based on sparsity
    if use_sparse and (sparse.issparse(adata.X) or adata.shape[0] > 50000):
        print("Using sparse-compatible TruncatedSVD...")
        
        # Convert to sparse if not already
        if sparse.issparse(adata.X):
            X = adata.X
        else:
            X = sparse.csr_matrix(adata.X)
        
        # Use TruncatedSVD for sparse matrices
        svd = TruncatedSVD(n_components=n_pcs, random_state=random_state)
        X_pca = svd.fit_transform(X)
        
        print(f"Explained variance ratio (first 10 PCs): {svd.explained_variance_ratio_[:10]}")
        print(f"Total variance explained by {n_pcs} PCs: {svd.explained_variance_ratio_.sum():.3f}")
        
        # Store PCA results
        adata.obsm['X_pca'] = X_pca
        adata.uns['pca'] = {
            'variance': svd.explained_variance_,
            'variance_ratio': svd.explained_variance_ratio_
        }
    else:
        print("Using standard PCA with scaling...")
        
        # Convert sparse to dense for standard scaling
        if sparse.issparse(adata.X):
            X_dense = adata.X.toarray()
        else:
            X_dense = adata.X
        
        # Standardize features
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X_dense)
        
        # Compute PCA
        pca = PCA(n_components=n_pcs, random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"PCA explained variance ratio (first 10 PCs): {pca.explained_variance_ratio_[:10]}")
        print(f"Total variance explained by {n_pcs} PCs: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Store PCA results
        adata.obsm['X_pca'] = X_pca
        adata.uns['pca'] = {
            'variance': pca.explained_variance_,
            'variance_ratio': pca.explained_variance_ratio_
        }
    
    print("\nComputing UMAP...")
    
    # Compute UMAP with low memory mode for large datasets
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        low_memory=True if adata.shape[0] > 50000 else False,
        verbose=True
    )
    
    X_umap = reducer.fit_transform(X_pca)
    
    # Store UMAP results
    adata.obsm['X_umap'] = X_umap
    
    print(f"UMAP coordinates shape: {X_umap.shape}")
    
    return adata


def plot_umap_by_groups(adata, save_prefix=None, figsize=(24, 5), save_dir='raw_data_check'):
    """
    Plot UMAP colored by modality, cell_type, batch_id, and meta_batch_id
    
    Args:
        adata: AnnData object with UMAP computed
        save_prefix: Prefix for saving figures (optional)
        figsize: Figure size (default: (24, 5))
        save_dir: Directory to save figures (default: 'raw_data_check')
    """
    if 'X_umap' not in adata.obsm:
        raise ValueError("UMAP not computed. Run compute_umap() first.")
    
    # Create save directory if it doesn't exist
    if save_prefix is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    umap_coords = adata.obsm['X_umap']
    
    # Print legend information for debugging
    print("\n" + "="*60)
    print("LEGEND INFORMATION FOR DEBUGGING")
    print("="*60)
    
    modality = adata.obs['modality']
    unique_modalities = sorted(modality.unique())
    print(f"\nModalities ({len(unique_modalities)}):")
    print(f"  {unique_modalities}")
    
    cell_types = adata.obs['cell_type']
    unique_cell_types = sorted(cell_types.unique())
    n_cell_types = len(unique_cell_types)
    print(f"\nCell Types ({n_cell_types}):")
    if n_cell_types <= 50:
        print(f"  {unique_cell_types}")
    else:
        print(f"  First 20: {unique_cell_types[:20]}")
        print(f"  Last 20: {unique_cell_types[-20:]}")
        print(f"  (showing first/last 20 of {n_cell_types} total)")
    
    batch_ids = adata.obs['batch_id_clean']  # Use cleaned batch IDs
    unique_batch_ids = sorted(batch_ids.unique())
    n_batches = len(unique_batch_ids)
    print(f"\nBatch IDs (cleaned) ({n_batches}):")
    if n_batches <= 50:
        print(f"  {unique_batch_ids}")
    else:
        print(f"  First 20: {unique_batch_ids[:20]}")
        print(f"  Last 20: {unique_batch_ids[-20:]}")
        print(f"  (showing first/last 20 of {n_batches} total)")
    
    meta_batch_ids = adata.obs['meta_batch_id']
    unique_meta_batch_ids = sorted(meta_batch_ids.unique())
    n_meta_batches = len(unique_meta_batch_ids)
    print(f"\nMeta Batch IDs ({n_meta_batches}):")
    print(f"  {unique_meta_batch_ids}")
    
    # Print counts per category
    print(f"\nCell counts per modality:")
    for mod in unique_modalities:
        count = (modality == mod).sum()
        print(f"  {mod}: {count:,} cells")
    
    print(f"\nTop 10 cell types by count:")
    cell_type_counts = cell_types.value_counts()
    for ct, count in cell_type_counts.head(10).items():
        print(f"  {ct}: {count:,} cells")
    
    print(f"\nTop 10 batches (cleaned) by count:")
    batch_counts = batch_ids.value_counts()
    for batch, count in batch_counts.head(10).items():
        print(f"  {batch}: {count:,} cells")
    
    print(f"\nMeta batch counts:")
    meta_batch_counts = meta_batch_ids.value_counts()
    for meta_batch, count in meta_batch_counts.items():
        print(f"  {meta_batch}: {count:,} cells")
    
    print("="*60 + "\n")
    
    # Create figure with four subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Plot 1: Color by modality
    print("Plot 1: Color by modality") 
    colors_modality = {'cell': '#1f77b4', 'nuclei': '#ff7f0e'}
    
    for mod in unique_modalities:
        mask = modality == mod
        axes[0].scatter(
            umap_coords[mask, 0], 
            umap_coords[mask, 1],
            c=colors_modality.get(mod, '#gray'),
            label=mod,
            alpha=0.6,
            s=1
        )
    
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].set_title('UMAP colored by Modality')
    axes[0].legend(markerscale=5, loc='upper right', framealpha=0.8)
    
    # Plot 2: Color by cell_type
    print("Plot 2: Color by cell_type")
    # Use tab20 for up to 20 cell types, extend with tab20b and tab20c if needed
    if n_cell_types <= 20:
        colors_ct = plt.cm.tab20(np.linspace(0, 1, n_cell_types))
    else:
        # Combine multiple colormaps for more categories
        colors_ct = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20b(np.linspace(0, 1, min(20, n_cell_types - 20))),
        ])
        if n_cell_types > 40:
            colors_ct = np.vstack([
                colors_ct,
                plt.cm.tab20c(np.linspace(0, 1, n_cell_types - 40))
            ])
    
    ct_to_color = dict(zip(unique_cell_types, colors_ct))
    
    for ct in unique_cell_types:
        mask = cell_types == ct
        axes[1].scatter(
            umap_coords[mask, 0], 
            umap_coords[mask, 1],
            c=[ct_to_color[ct]],
            label=ct,
            alpha=0.6,
            s=1
        )
    
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].set_title(f'UMAP colored by Cell Type (n={n_cell_types})')
    
    # Only show legend if reasonable number of categories
    if n_cell_types <= 20:
        axes[1].legend(markerscale=3, loc='upper right', fontsize=6, framealpha=0.8)
    else:
        # Too many cell types, skip legend
        axes[1].text(0.95, 0.95, f'{n_cell_types} cell types', 
                    transform=axes[1].transAxes, 
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: Color by batch_id (cleaned)
    print("Plot 3: Color by batch_id (cleaned)")
    if n_batches <= 20:
        colors_batch = plt.cm.tab20(np.linspace(0, 1, n_batches))
    else:
        colors_batch = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20b(np.linspace(0, 1, min(20, n_batches - 20))),
        ])
        if n_batches > 40:
            colors_batch = np.vstack([
                colors_batch,
                plt.cm.tab20c(np.linspace(0, 1, n_batches - 40))
            ])
    
    batch_to_color = dict(zip(unique_batch_ids, colors_batch))
    
    for batch in unique_batch_ids:
        mask = batch_ids == batch
        axes[2].scatter(
            umap_coords[mask, 0], 
            umap_coords[mask, 1],
            c=[batch_to_color[batch]],
            label=batch,
            alpha=0.6,
            s=1
        )
    
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')
    axes[2].set_title(f'UMAP colored by Batch ID (n={n_batches})')
    
    # Only show legend if reasonable number of categories
    if n_batches <= 20:
        axes[2].legend(markerscale=3, loc='upper right', fontsize=6, framealpha=0.8)
    else:
        # Too many batches, skip legend
        axes[2].text(0.95, 0.95, f'{n_batches} batches', 
                    transform=axes[2].transAxes, 
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: Color by meta_batch_id
    print("Plot 4: Color by meta_batch_id")
    if n_meta_batches <= 20:
        colors_meta = plt.cm.tab10(np.linspace(0, 1, n_meta_batches))
    else:
        colors_meta = plt.cm.tab20(np.linspace(0, 1, n_meta_batches))
    
    meta_to_color = dict(zip(unique_meta_batch_ids, colors_meta))
    
    for meta_batch in unique_meta_batch_ids:
        mask = meta_batch_ids == meta_batch
        axes[3].scatter(
            umap_coords[mask, 0], 
            umap_coords[mask, 1],
            c=[meta_to_color[meta_batch]],
            label=meta_batch,
            alpha=0.6,
            s=1
        )
    
    axes[3].set_xlabel('UMAP 1')
    axes[3].set_ylabel('UMAP 2')
    axes[3].set_title(f'UMAP colored by Meta Batch ID (n={n_meta_batches})')
    axes[3].legend(markerscale=3, loc='upper right', fontsize=7, framealpha=0.8)
    
    plt.tight_layout()
    
    if save_prefix:
        save_path = os.path.join(save_dir, f'{save_prefix}_umap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        # Save separate legend files if there are many categories
        if n_cell_types > 20 or n_batches > 20:
            print("Saving separate legend files for large categorical variables...")
            
            # Cell type legend
            if n_cell_types > 20:
                fig_legend_ct = plt.figure(figsize=(8, max(10, n_cell_types * 0.3)))
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=ct_to_color[ct], 
                                              markersize=8, label=ct) 
                                  for ct in sorted(unique_cell_types)]
                fig_legend_ct.legend(handles=legend_elements, loc='center', 
                                    ncol=max(1, n_cell_types // 40), fontsize=8)
                fig_legend_ct.tight_layout()
                legend_path_ct = os.path.join(save_dir, f'{save_prefix}_cell_type_legend.png')
                fig_legend_ct.savefig(legend_path_ct, dpi=150, bbox_inches='tight')
                plt.close(fig_legend_ct)
                print(f"  Cell type legend saved: {legend_path_ct}")
            
            # Batch ID legend
            if n_batches > 20:
                fig_legend_batch = plt.figure(figsize=(8, max(10, n_batches * 0.3)))
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=batch_to_color[batch], 
                                              markersize=8, label=batch) 
                                  for batch in sorted(unique_batch_ids)]
                fig_legend_batch.legend(handles=legend_elements, loc='center', 
                                       ncol=max(1, n_batches // 40), fontsize=8)
                fig_legend_batch.tight_layout()
                legend_path_batch = os.path.join(save_dir, f'{save_prefix}_batch_legend.png')
                fig_legend_batch.savefig(legend_path_batch, dpi=150, bbox_inches='tight')
                plt.close(fig_legend_batch)
                print(f"  Batch ID legend saved: {legend_path_batch}")
    
    plt.show()
    """
    Plot UMAP colored by modality, cell_type, and batch_id
    
    Args:
        adata: AnnData object with UMAP computed
        save_prefix: Prefix for saving figures (optional)
        figsize: Figure size (default: (18, 5))
        save_dir: Directory to save figures (default: 'raw_data_check')
    """
    if 'X_umap' not in adata.obsm:
        raise ValueError("UMAP not computed. Run compute_umap() first.")
    
    # Create save directory if it doesn't exist
    if save_prefix is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    umap_coords = adata.obsm['X_umap']
    
    # Print legend information for debugging
    print("\n" + "="*60)
    print("LEGEND INFORMATION FOR DEBUGGING")
    print("="*60)
    
    modality = adata.obs['modality']
    unique_modalities = sorted(modality.unique())
    print(f"\nModalities ({len(unique_modalities)}):")
    print(f"  {unique_modalities}")
    
    cell_types = adata.obs['cell_type']
    unique_cell_types = sorted(cell_types.unique())
    n_cell_types = len(unique_cell_types)
    print(f"\nCell Types ({n_cell_types}):")
    if n_cell_types <= 50:
        print(f"  {unique_cell_types}")
    else:
        print(f"  First 20: {unique_cell_types[:20]}")
        print(f"  Last 20: {unique_cell_types[-20:]}")
        print(f"  (showing first/last 20 of {n_cell_types} total)")
    
    batch_ids = adata.obs['batch_id_clean']  # Use cleaned batch IDs
    unique_batch_ids = sorted(batch_ids.unique())
    n_batches = len(unique_batch_ids)
    print(f"\nBatch IDs (cleaned) ({n_batches}):")
    if n_batches <= 50:
        print(f"  {unique_batch_ids}")
    else:
        print(f"  First 20: {unique_batch_ids[:20]}")
        print(f"  Last 20: {unique_batch_ids[-20:]}")
        print(f"  (showing first/last 20 of {n_batches} total)")
    
    # Print counts per category
    print(f"\nCell counts per modality:")
    for mod in unique_modalities:
        count = (modality == mod).sum()
        print(f"  {mod}: {count:,} cells")
    
    print(f"\nTop 10 cell types by count:")
    cell_type_counts = cell_types.value_counts()
    for ct, count in cell_type_counts.head(10).items():
        print(f"  {ct}: {count:,} cells")
    
    print(f"\nTop 10 batches (cleaned) by count:")
    batch_counts = batch_ids.value_counts()
    for batch, count in batch_counts.head(10).items():
        print(f"  {batch}: {count:,} cells")
    
    print("="*60 + "\n")
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Color by modality
    colors_modality = {'cell': '#1f77b4', 'nuclei': '#ff7f0e'}
    
    for mod in unique_modalities:
        mask = modality == mod
        axes[0].scatter(
            umap_coords[mask, 0], 
            umap_coords[mask, 1],
            c=colors_modality.get(mod, '#gray'),
            label=mod,
            alpha=0.6,
            s=1
        )
    
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].set_title('UMAP colored by Modality')
    axes[0].legend(markerscale=5, loc='upper right', framealpha=0.8)
    
    # Plot 2: Color by cell_type
    cell_types = adata.obs['cell_type']
    unique_cell_types = cell_types.unique()
    
    # Use tab20 for up to 20 cell types, extend with tab20b and tab20c if needed
    n_cell_types = len(unique_cell_types)
    if n_cell_types <= 25:
        colors_ct = plt.cm.tab20(np.linspace(0, 1, n_cell_types))
    else:
        # Combine multiple colormaps for more categories
        colors_ct = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20b(np.linspace(0, 1, min(20, n_cell_types - 20))),
        ])
        if n_cell_types > 40:
            colors_ct = np.vstack([
                colors_ct,
                plt.cm.tab20c(np.linspace(0, 1, n_cell_types - 40))
            ])
    
    ct_to_color = dict(zip(unique_cell_types, colors_ct))
    
    for ct in unique_cell_types:
        mask = cell_types == ct
        axes[1].scatter(
            umap_coords[mask, 0], 
            umap_coords[mask, 1],
            c=[ct_to_color[ct]],
            label=ct,
            alpha=0.6,
            s=1
        )
    
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].set_title(f'UMAP colored by Cell Type (n={n_cell_types})')
    
    # Only show legend if reasonable number of categories
    if n_cell_types <= 25:
        axes[1].legend(markerscale=3, loc='upper right', fontsize=6, framealpha=0.8)
    else:
        # Too many cell types, skip legend
        axes[1].text(0.95, 0.95, f'{n_cell_types} cell types', 
                    transform=axes[1].transAxes, 
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: Color by batch_id
    batch_ids = adata.obs['batch_id_clean']
    unique_batch_ids = batch_ids.unique()
    n_batches = len(unique_batch_ids)
    
    if n_batches <= 25:
        colors_batch = plt.cm.tab20(np.linspace(0, 1, n_batches))
    else:
        colors_batch = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20b(np.linspace(0, 1, min(20, n_batches - 20))),
        ])
        if n_batches > 40:
            colors_batch = np.vstack([
                colors_batch,
                plt.cm.tab20c(np.linspace(0, 1, n_batches - 40))
            ])
    
    batch_to_color = dict(zip(unique_batch_ids, colors_batch))
    
    for batch in unique_batch_ids:
        mask = batch_ids == batch
        axes[2].scatter(
            umap_coords[mask, 0], 
            umap_coords[mask, 1],
            c=[batch_to_color[batch]],
            label=batch,
            alpha=0.6,
            s=1
        )
    
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')
    axes[2].set_title(f'UMAP colored by Batch ID (n={n_batches})')
    
    # Only show legend if reasonable number of categories
    if n_batches <= 25:
        axes[2].legend(markerscale=3, loc='upper right', fontsize=6, framealpha=0.8)
    else:
        # Too many batches, skip legend
        axes[2].text(0.95, 0.95, f'{n_batches} batches', 
                    transform=axes[2].transAxes, 
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_prefix:
        save_path = os.path.join(save_dir, f'{save_prefix}_umap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        # Save separate legend files if there are many categories
        if n_cell_types > 20 or n_batches > 20:
            print("Saving separate legend files for large categorical variables...")
            
            # Cell type legend
            if n_cell_types > 25:
                fig_legend_ct = plt.figure(figsize=(8, max(10, n_cell_types * 0.3)))
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=ct_to_color[ct], 
                                              markersize=8, label=ct) 
                                  for ct in sorted(unique_cell_types)]
                fig_legend_ct.legend(handles=legend_elements, loc='center', 
                                    ncol=max(1, n_cell_types // 40), fontsize=8)
                fig_legend_ct.tight_layout()
                legend_path_ct = os.path.join(save_dir, f'{save_prefix}_cell_type_legend.png')
                fig_legend_ct.savefig(legend_path_ct, dpi=150, bbox_inches='tight')
                plt.close(fig_legend_ct)
                print(f"  Cell type legend saved: {legend_path_ct}")
            
            # Batch ID legend
            if n_batches > 25:
                fig_legend_batch = plt.figure(figsize=(8, max(10, n_batches * 0.3)))
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=batch_to_color[batch], 
                                              markersize=8, label=batch) 
                                  for batch in sorted(unique_batch_ids)]
                fig_legend_batch.legend(handles=legend_elements, loc='center', 
                                       ncol=max(1, n_batches // 40), fontsize=8)
                fig_legend_batch.tight_layout()
                legend_path_batch = os.path.join(save_dir, f'{save_prefix}_batch_legend.png')
                fig_legend_batch.savefig(legend_path_batch, dpi=150, bbox_inches='tight')
                plt.close(fig_legend_batch)
                print(f"  Batch ID legend saved: {legend_path_batch}")
    
    plt.show()


def select_by_meta_batch_id(cells_adata, nuclei_adata, selected_meta_batch_ids, 
                            use_log1p=True, n_top_genes=None):
    """
    Select data based on a list of meta_batch_ids from both cell and nuclei datasets
    
    Args:
        cells_adata: AnnData object for cells
        nuclei_adata: AnnData object for nuclei
        selected_meta_batch_ids: List of meta_batch_ids to select (e.g., ['n3A', 'c3A'])
        use_log1p: Whether to use log1p layer (default: True)
        n_top_genes: Number of highly variable genes to select (optional)
    
    Returns:
        Combined AnnData object with selected meta_batch_ids
    """
    # Process batch IDs
    cells_adata = process_batch_ids(cells_adata)
    nuclei_adata = process_batch_ids(nuclei_adata)
    
    # Add modality labels
    cells_adata.obs['modality'] = 'cell'
    nuclei_adata.obs['modality'] = 'nuclei'
    
    print(f"\nSelecting meta_batch_ids: {selected_meta_batch_ids}")
    print("=" * 60)
    
    # Find which meta_batch_ids exist in each dataset
    cells_meta_batches = set(cells_adata.obs['meta_batch_id'].unique())
    nuclei_meta_batches = set(nuclei_adata.obs['meta_batch_id'].unique())
    
    print(f"Cell dataset contains meta_batch_ids: {sorted(cells_meta_batches)}")
    print(f"Nuclei dataset contains meta_batch_ids: {sorted(nuclei_meta_batches)}")
    
    # Find intersection with requested IDs
    cells_selected = set(selected_meta_batch_ids) & cells_meta_batches
    nuclei_selected = set(selected_meta_batch_ids) & nuclei_meta_batches
    
    print(f"\nFrom cells: {sorted(cells_selected) if cells_selected else 'None'}")
    print(f"From nuclei: {sorted(nuclei_selected) if nuclei_selected else 'None'}")
    
    # Filter datasets
    selected_datasets = []
    
    if cells_selected:
        cells_filtered = cells_adata[cells_adata.obs['meta_batch_id'].isin(cells_selected)].copy()
        print(f"\nCells filtered: {cells_filtered.shape[0]} cells")
        selected_datasets.append(cells_filtered)
    else:
        print("\nNo matching meta_batch_ids found in cells dataset")
    
    if nuclei_selected:
        nuclei_filtered = nuclei_adata[nuclei_adata.obs['meta_batch_id'].isin(nuclei_selected)].copy()
        print(f"Nuclei filtered: {nuclei_filtered.shape[0]} nuclei")
        selected_datasets.append(nuclei_filtered)
    else:
        print("No matching meta_batch_ids found in nuclei dataset")
    
    # Handle case where no data is found
    if not selected_datasets:
        print("\nWARNING: No data found for the selected meta_batch_ids!")
        return None
    
    # If only one dataset has data, return it directly
    if len(selected_datasets) == 1:
        print(f"\nOnly one modality has data. Returning {selected_datasets[0].obs['modality'].iloc[0]} dataset only.")
        selected_adata = selected_datasets[0]
    else:
        # Find common genes
        common_genes = selected_datasets[0].var_names.intersection(selected_datasets[1].var_names)
        print(f"\nCommon genes: {len(common_genes)}")
        
        # Subset to common genes
        for i in range(len(selected_datasets)):
            selected_datasets[i] = selected_datasets[i][:, common_genes].copy()
            
            # Use log1p layer if specified
            if use_log1p and 'log1p' in selected_datasets[i].layers:
                selected_datasets[i].X = selected_datasets[i].layers['log1p']
        
        # Concatenate
        selected_adata = ad.concat(
            selected_datasets,
            join='outer',
            index_unique='_'
        )
    
    print(f"\nFinal combined dataset shape: {selected_adata.shape}")
    print(f"Modality distribution:\n{selected_adata.obs['modality'].value_counts()}")
    print(f"Meta batch ID distribution:\n{selected_adata.obs['meta_batch_id'].value_counts()}")
    
    # Select highly variable genes if specified
    if n_top_genes is not None:
        selected_adata = select_highly_variable_genes(selected_adata, n_top_genes=n_top_genes)
    
    return selected_adata


# Example usage
if __name__ == "__main__":
    # Load data
    print("Loading datasets...")
    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nuclei_ann_save_dir = os.path.join(_project_dir, 'data', 'scvi', '10X_nuclei_v2_v3_AIBS', 'data.h5ad')
    cells_ann_save_dir = os.path.join(_project_dir, 'data', 'scvi', '10X_cells_v2_v3_AIBS', 'data.h5ad')
    
    nuclei_adata = ad.read_h5ad(nuclei_ann_save_dir)
    cells_adata = ad.read_h5ad(cells_ann_save_dir)
    
    # Add log1p layers if not already present
    if 'log1p' not in nuclei_adata.layers:
        nuclei_adata.layers['log1p'] = np.log1p(nuclei_adata.X)
    if 'log1p' not in cells_adata.layers:
        cells_adata.layers['log1p'] = np.log1p(cells_adata.X)
    
    # Example 1: Combine all data with memory-efficient settings
    print("\n" + "="*60)
    print("EXAMPLE 1: Memory-efficient processing of all data")
    print("="*60)
    combined_adata = combine_cell_nuclei_data(
        cells_adata, 
        nuclei_adata,
        n_top_genes=3000  # Use only 3000 most variable genes
    )
    combined_adata = compute_umap(
        combined_adata,
        n_pcs=30,  # Reduce number of PCs
        n_neighbors=15,
        max_cells=50000,  # Downsample to 50k cells for visualization
        use_sparse=True  # Use sparse-compatible methods
    )
    plot_umap_by_groups(combined_adata, save_prefix='all_data', save_dir='raw_data_check')
    
    # Example 2: Select specific meta_batch_ids
    print("\n" + "="*60)
    print("EXAMPLE 2: Selecting specific meta_batch_ids")
    print("="*60)
    selected_adata = select_by_meta_batch_id(
        cells_adata, 
        nuclei_adata, 
        selected_meta_batch_ids=['n3A', 'c3A'],
        n_top_genes=3000
    )
    
    if selected_adata is not None:
        selected_adata = compute_umap(
            selected_adata,
            n_pcs=30,
            n_neighbors=15,
            max_cells=30000,
            use_sparse=True
        )
        plot_umap_by_groups(selected_adata, save_prefix='selected_data', save_dir='raw_data_check')
    
    # Example 3: Process very large dataset with aggressive downsampling
    print("\n" + "="*60)
    print("EXAMPLE 3: Aggressive downsampling for quick visualization")
    print("="*60)
    quick_adata = combine_cell_nuclei_data(
        cells_adata, 
        nuclei_adata,
        n_top_genes=2000
    )
    quick_adata = compute_umap(
        quick_adata,
        n_pcs=20,
        n_neighbors=10,
        max_cells=20000,  # Very aggressive downsampling
        use_sparse=True
    )
    plot_umap_by_groups(quick_adata, save_prefix='quick_viz', save_dir='raw_data_check')
