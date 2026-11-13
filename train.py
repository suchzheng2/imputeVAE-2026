import sys
import os
import subprocess
import numpy as np
import pandas as pd
import math
import time
import random
from copy import deepcopy
import argparse
import multiprocessing as mp
from functools import partial
import glob
import json
import csv
import imputeVAE as ConditionalVAE
import datetime
import anndata as ad
# from pegasusio import UnimodalData, MultimodalData


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, set):
        return {convert_to_serializable(item) for item in obj}
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.ndarray, np.bool_)):
        return obj.tolist()
    else:
        return obj

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_internal_size(arg):
    # If arg is something like "1024,512,256"
    if "," in arg:
        return [2 ** int(x) for x in arg.split(",")]
    # Otherwise interpret as exponent
    else:
        return [2 ** int(arg)]
        
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Takes in argument for VAE')
    
    #required argument
    parser.add_argument(
        "--model",
        default="vae",
        type=str,
        required=False,
        help="Model type: vae, vae_l1, vae_sl, vae_JL, vae_nb (Negative Binomial for raw counts)",
    )

    parser.add_argument(
        "--data_path",
        "-dp",
        default=None,
        type=str,
        required=True,
        help="path to ABC output: integreated.h5ad",
    )

    parser.add_argument(
        "--batch",
        "-b",
        default=None,
        type=str,
        required=False,
        help="batch n2A or batch n3A",
    )

    parser.add_argument(
        "--protocol",
        default=None,
        type=str,
        choices=["v2", "v3", "both"],
        required=False,
        help="10X protocol version: v2, v3, or both. "
             "Overrides --batch filtering to select by protocol version. "
             "e.g. --protocol v2 with nuclei data selects all n2A batches, "
             "--protocol both selects both n2A and n3A.",
    )


    parser.add_argument(
        "--norm",
        default="minmax",
        type=str,
        required=False,
        help="normalization method, minmax",
    )

    parser.add_argument(
        "--internal_size",
        default="7",
        type=str,
        help="Hidden layer sizes. Either single int exponent N (2^N), "
            "or comma-separated list, e.g. '1024,512,256'"
    )

    parser.add_argument(
        "--latent_size",
        default=7,
        type=int,
        required=False,
        help="VAE latent size, 2^N",
    )

    parser.add_argument(
        "--num_feature",
        default=0,
        type=int,
        required=False,
        help="number of feature",
    )

    parser.add_argument(
        "--num_latent",
        default=0,
        type=int,
        required=False,
        help="number of latent node, 2^latent_size",
    )

    parser.add_argument(
        "--num_internal",
        default=0,
        type=int,
        required=False,
        help="number of internal node, 2^internal_size",
    )

    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        required=False,
        help="number of internal node, 2^internal_size",
    )

    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        required=False,
        help="start epoch, if use this, specify the loaded model path",
    )

    parser.add_argument(
        "--num_epochs",
        default=10,
        type=int,
        required=False,
        help="number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        required=False,
        help="number of data samples in one batch",
    )

    parser.add_argument(
        "--beta",
        default=1,
        type=float,
        required=False,
        help="weight gives to KL_div, range(0,1)",
    )

    parser.add_argument(
        "--l1_coef",
        default=0.01,
        type=float,
        required=False,
        help="if use vae_l1, this argument is required",
    )

    parser.add_argument(
        "--valid",
        default=True,
        type=bool,
        required=False,
        help="validate model after each epoch",
    )

    parser.add_argument(
        "--test",
        default=True,
        type=bool,
        required=False,
        help="test model after all epochs",
    )

    parser.add_argument(
        "--testCode",
        default=False,
        type=bool,
        required=False,
        help="run test loop",
    )

    parser.add_argument(
        "--recons_with_log",
        default=True,
        type=str2bool,
        required=False,
        help="** old argment, not in use. True equals to recons_with_log_sqrt, false equals to recons_with_sqrt",
    )

    parser.add_argument(
        "--recons_loss_type",
        default=0,
        type=int,
        required=False,
        help="what kind of reconstruction loss to use, 0 - MSE, 1-MSE with sqrt, 2-MSE with sqrt and log)"
    )

    parser.add_argument(
        "--clamp_recons_loss",
        default=True,
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        required=False,
        help="Clamp reconstruction loss to [1e-6, 1-1e-6]. Disable for CPM or other normalizations where MSE > 1."
    )


    parser.add_argument(
        "--limit_l1",
        default=False,
        type=str2bool,
        required=False,
        help="limit L1 on first layer or not",
    )

    parser.add_argument(
        "--avg_l1",
        default=False,
        type=str2bool,
        required=False,
        help="avgerage L1 among all weights",
    )

    # if want to test model e.g. test cell model on nuclei dataset
    parser.add_argument(
        "--test_model",
        default=False,
        type=str2bool,
        required=False,
        help="Test model on another datatset",
    )

    parser.add_argument(
        "--test_model_run",
        default=None,
        type=str,
        required=False,
        help="The target model's run path, ./run/run_XXX",
    )

    parser.add_argument(
        "--test_data",
        default=None,
        type=str,
        required=False,
        help="dataloader of the test dataset",
    )

    parser.add_argument(
        "--add_mu_clamp",
        default=None,
        type=float,
        required=False,
        help="add a clamp on mu",
    )

    parser.add_argument(
        "--add_log_var_clamp",
        default=None,
        type=float,
        required=False,
        help="add a clamp on log_var",
    )

    #set up anneal beta

    parser.add_argument(
        "--kl_anneal", 
        type=str2bool, 
        default=False, 
        help="Enable KL annealing"
    )

    parser.add_argument(
        "--kl_anneal_start_epoch", 
        type=int, 
        default=0, 
        help="Epochs start to add beta"
    )

    parser.add_argument(
        "--kl_anneal_epoch", 
        type=int, 
        default=20, 
        help="Epochs to reach full beta"
    )


    parser.add_argument("--use_soft_constraint", type=str2bool, default=False,
                    help="Enable soft constraint penalty on mu/log_var")
    parser.add_argument("--soft_constraint_lambda", type=float, default=1e-3,
                    help="Lambda coefficient for soft constraint regularization")

    parser.add_argument("--normalize_losses", type=str2bool, default=False,
                    help="Scale recon loss by epoch-0 value and JSD by ln(2) so both are ~[0,1]")
    parser.add_argument("--recon_loss_norm", type=float, default=0.0,
                    help="Recon loss normalization constant; 0 = auto-compute from epoch 0")

    parser.add_argument("--kl_clamp", type=float, default=None,
                    help="kl_clamp to regularized KL term not smaller this number")
    
    parser.add_argument("--no_lr", type=str2bool, default=False,
                    help="Let Adam define the learning rate")

    parser.add_argument("--subset", type=str2bool, default=False,
                    help="use full dataser for training as default, but if this argument is true, will be using the subset of the data")
    parser.add_argument("--subset_ratio", type=float, default=0.1,
                    help="portion of data used")
    
    parser.add_argument("--plot_umap_after_train", type=str2bool, default=False,
                    help="plot umap or not")
    
    parser.add_argument("--norm_KL", type=str2bool, default=False,
                    help="normalize KL by recons_x dim")
    # JS Div                
    parser.add_argument("--JS_dual", type=str2bool, default=False,
                    help="Whether to use Dual or standard GJS.")    
    parser.add_argument("--JS_a", type=float, default=0.5,
                    help="Skew of the skew geometric-Jensen-Shannon divergence")  
    parser.add_argument("--JS_invert_alpha", type=str2bool, default=False,
                    help="Whether to replace alpha with 1 - a.")           
    parser.add_argument("--JS_norm", type=str2bool, default=False,
                    help="Whether to replace alpha with 1 - a.")      
    

    # test combined model
    parser.add_argument("--test_combined", type=str2bool, default=False,
                    help="True if we are testing the combine model with encoder trained on nuclei and decoder trained on cell")
    parser.add_argument("--test_combined_mixed_modality", type=str2bool, default=False,
                    help="True if we are testing the combine model with encoder trained on nuclei and decoder trained on cell")
    parser.add_argument("--nuclei_model_run", default=None, type=str, required=False,
                    help="run file to the nuclei model")
    parser.add_argument("--cell_model_run", default=None, type=str, required=False,
                    help="run file to the cell model")

    # fine tune model mode:
    parser.add_argument("--fine_tune_model", type=str2bool, default=False,
                    help="True if we are fine tuning the model, use --nuclei_model_run and --cell_model_run to specify the model to fine tune")
    parser.add_argument("--alpha", type=float, default=0.5,
                    help="Legacy alignment weight for fine tuning; stage3_v2 maps this to --alignment_weight")
    # anneal alpha:
    parser.add_argument("--anneal_alpha", type=str2bool, default=False,
                    help="Legacy alignment annealing flag; stage3_v2 maps this to --anneal_alignment")
    parser.add_argument("--anneal_alpha_start_epoch", type=int, default=0,
                    help="Legacy alignment anneal start; stage3_v2 maps this to --anneal_alignment_start_epoch")
    parser.add_argument("--anneal_alpha_epoch", type=int, default=20,
                    help="Legacy alignment anneal length; stage3_v2 maps this to --anneal_alignment_epoch")
    
    parser.add_argument("--ft_metrics", type=str, default="jsd",
                    help="metrics for the fine tune model")
    parser.add_argument("--was_train_by_cell_type", type=str2bool, default=False,
                    help="True if the cell model was trained by cell type")
    parser.add_argument("--use_all_cell_mu_log_var", type=str2bool, default=False,
                    help="True if the cell model was trained by all cell types")

    

    # debug
    parser.add_argument("--debug", type=str2bool, default=False,
                    help="True if we are debugging the model")
    parser.add_argument("--run_debug_function", type=str2bool, default=False,
                    help="True if we are running the debug function")

    parser.add_argument("--test_nuclei_model_on_cell_data", type=str2bool, default=False,
                    help="True if we are testing the nuclei model on the cell data")

    # continue training;
    parser.add_argument("--continue_training", type=str2bool, default=False,
                    help="start epoch for continue training")
    parser.add_argument("--continue_training_run_path", type=str, default=None,
                    help="path to the trained model")
    parser.add_argument("--continue_training_epoch", type=int, default=0,
                    help="epoch for continue training")
    
    # fine tune by stage:
    parser.add_argument("--load_trained_model", type=str2bool, default=False,
                    help="True if we are loading the trained model")
    parser.add_argument("--load_trained_model_path", type=str, default=None,
                    help="Path to the trained model")
    parser.add_argument("--fine_tune_stage3", type=str2bool, default=False,
                    help="True if we are fine tuning the model for stage 3")
    parser.add_argument("--cell_recons_weight", type=float, default=1.0,
                    help="weight for the cell reconstruction loss")
    parser.add_argument("--cell_jsd_weight", type=float, default=1.0,
                    help="weight for the cell JSD loss during joint cross-modality training")
    parser.add_argument("--nuclei_recons_weight", type=float, default=1.0,
                    help="weight for the nuclei reconstruction loss during joint cross-modality training")
    parser.add_argument("--nuclei_jsd_weight", type=float, default=1.0,
                    help="weight for the nuclei JSD loss during joint cross-modality training")
    parser.add_argument("--alignment_weight", type=float, default=None,
                    help="explicit weight for the cross-modality alignment JSD term in stage3_v2")
    parser.add_argument("--anneal_cell_recons_weight", type=str2bool, default=False,
                    help="anneal the cell reconstruction weight from 0 to its configured value")
    parser.add_argument("--cell_recons_weight_start_epoch", type=int, default=0,
                    help="epoch to start annealing cell reconstruction weight")
    parser.add_argument("--cell_recons_weight_anneal_epoch", type=int, default=20,
                    help="number of epochs to reach the final cell reconstruction weight")
    parser.add_argument("--anneal_cell_jsd_weight", type=str2bool, default=False,
                    help="anneal the cell prior-JSD weight from 0 to its configured value")
    parser.add_argument("--cell_jsd_weight_start_epoch", type=int, default=0,
                    help="epoch to start annealing cell prior-JSD weight")
    parser.add_argument("--cell_jsd_weight_anneal_epoch", type=int, default=20,
                    help="number of epochs to reach the final cell prior-JSD weight")
    parser.add_argument("--anneal_nuclei_recons_weight", type=str2bool, default=False,
                    help="anneal the nuclei reconstruction weight from 0 to its configured value")
    parser.add_argument("--nuclei_recons_weight_start_epoch", type=int, default=0,
                    help="epoch to start annealing nuclei reconstruction weight")
    parser.add_argument("--nuclei_recons_weight_anneal_epoch", type=int, default=20,
                    help="number of epochs to reach the final nuclei reconstruction weight")
    parser.add_argument("--anneal_nuclei_jsd_weight", type=str2bool, default=False,
                    help="anneal the nuclei prior-JSD weight from 0 to its configured value")
    parser.add_argument("--nuclei_jsd_weight_start_epoch", type=int, default=0,
                    help="epoch to start annealing nuclei prior-JSD weight")
    parser.add_argument("--nuclei_jsd_weight_anneal_epoch", type=int, default=20,
                    help="number of epochs to reach the final nuclei prior-JSD weight")
    parser.add_argument("--anneal_alignment", type=str2bool, default=False,
                    help="anneal the cross-modality alignment weight from 0 to its configured value")
    parser.add_argument("--anneal_alignment_start_epoch", type=int, default=0,
                    help="epoch to start annealing alignment weight")
    parser.add_argument("--anneal_alignment_epoch", type=int, default=20,
                    help="number of epochs to reach the final alignment weight")

    # cross-modality reconstruction loss (decoded nuclei → raw cell)
    parser.add_argument("--cross_recons_weight", type=float, default=0.0,
                    help="weight for cross-modality reconstruction loss (decoded nuclei vs raw cell); 0 disables")
    parser.add_argument("--anneal_cross_recons_weight", type=str2bool, default=False,
                    help="anneal the cross-reconstruction weight from 0 to its configured value")
    parser.add_argument("--cross_recons_weight_start_epoch", type=int, default=0,
                    help="epoch to start annealing cross-reconstruction weight")
    parser.add_argument("--cross_recons_weight_anneal_epoch", type=int, default=100,
                    help="number of epochs to reach the final cross-reconstruction weight")

    # residual MSE loss (per-cell deviation reconstruction)
    parser.add_argument("--residual_weight", type=float, default=0.0,
                    help="weight for residual MSE loss (MSE on deviations from CT mean); 0 disables")
    parser.add_argument("--anneal_residual_weight", type=str2bool, default=False,
                    help="anneal the residual MSE weight from 0 to its configured value")
    parser.add_argument("--residual_weight_start_epoch", type=int, default=0,
                    help="epoch to start annealing residual MSE weight")
    parser.add_argument("--residual_weight_anneal_epoch", type=int, default=100,
                    help="number of epochs to reach the final residual MSE weight")

    parser.add_argument("--fine_tune_stage2_v2", type=str2bool, default=False,
                    help="True if we are fine tuning the model for stage 2 v2")

    # stage 3 v2
    parser.add_argument("--fine_tune_stage3_v2", type=str2bool, default=False,
                    help="True if we are fine tuning the model for stage 3 v2")
    parser.add_argument("--stage1_model_run", type=str, default=None,
                    help="Path to the stage 1 model")
    parser.add_argument("--stage2_model_run", type=str, default=None,
                    help="Path to the stage 2 model")
    parser.add_argument("--stage3_eval_every", type=int, default=50,
                    help="Run compact inline stage3 evaluation every N epochs")
    parser.add_argument("--stage3_save_epoch_dump", type=str2bool, default=False,
                    help="Persist full stage3 epoch dumps; disabled by default to reduce storage use")
    parser.add_argument("--stage3_dump_every", type=int, default=50,
                    help="Save epoch dump every N epochs (only when --stage3_save_epoch_dump True)")
    parser.add_argument("--stage3_skip_full_eval", type=str2bool, default=False,
                    help="Skip full evaluation (heatmaps, correlation, scIB) at eval epochs; compact inline eval still runs")

    # use raw data for training:
    parser.add_argument("--use_raw_data", type=str2bool, default=False,
                    help="True if we are using the raw data for training")
    parser.add_argument("--test_model_on_whole_dataset", type=str2bool, default=False,
                    help="True if we are testing the model on the whole dataset")

    # for row data what normalization to use:
    parser.add_argument("--raw_data_normalization", type=str, default="log1p",
                    help="log1p, cpm, raw")

    # NB/ZINB model options:
    parser.add_argument("--use_zinb", type=str2bool, default=False,
                    help="Enable Zero-Inflated Negative Binomial (only used with --model vae_nb)")

    # Modality-conditioned decoder
    parser.add_argument("--fine_tune_cond_stage3", type=str2bool, default=False,
                    help="Train modality-conditioned VAE (VAE_NB_Cond) with decoder embedding")
    parser.add_argument("--embed_dim", type=int, default=8,
                    help="Dimension of modality embedding for conditioned decoder")
    parser.add_argument("--scalar_cond", type=str2bool, default=False,
                    help="Use fixed scalar conditioning (1=cell, 0=nuclei) instead of learned embedding")
    parser.add_argument("--cond_encoder", type=str2bool, default=False,
                    help="Also condition the encoder on modality (full cVAE); default conditions decoder only")

    # for loading raw data, apply common genes and cell type set
    parser.add_argument("--detection_rate_threshold", type=float, default=1.0,
                    help="detection rate threshold")
    parser.add_argument("--exclude_mitocarta", type=str2bool, default=False,
                    help="exclude mitocarta genes from adata")
    parser.add_argument("--gene_list", type=str, default=None,
                    help="path to gene list file (one gene per line); overrides detection_rate and mitocarta filters")
    parser.add_argument("--seed", type=int, default=None,
                    help="seed for weight init / shuffling (random, numpy, torch). Does NOT change the data split, "
                         "which is fixed at random_state=42. None = unseeded (legacy behaviour).")
    parser.add_argument("--libsize_cond", type=str2bool, default=False,
                    help="VAE_JSD_Cond only: feed standardised log1p(library size) to encoder and decoder as an "
                         "explicit covariate (notes/libsize_covariate_review_2026-10-09.md). Stats saved to "
                         "<run>/libsize_stats.json.")
    parser.add_argument("--libsize_target", type=str, default="ct_median_cell",
                    choices=["ct_median_cell", "source"],
                    help="libsize_cond only: library the inline cross-decode eval decodes nuclei at. "
                         "ct_median_cell = per-type median training-cell library (default); source = the nucleus's own.")
    parser.add_argument("--library_cond", type=str2bool, default=False,
                    help="VAE_JSD_Cond only: learned embedding of the sequencing library (meta_batch_id) fed to "
                         "encoder and decoder (2026-11-06). Vocabulary of training libraries saved to "
                         "<run>/library_vocab.json; id 0 = UNKNOWN (marginal library, used for unseen libraries "
                         "and as the default cross-decode target).")
    parser.add_argument("--library_dim", type=int, default=4, help="library_cond: embedding dimension")
    parser.add_argument("--library_capacity", type=int, default=32,
                    help="library_cond: embedding table size (max number of libraries; fixes state_dict shape)")
    parser.add_argument("--library_dropout", type=float, default=0.1,
                    help="library_cond: probability of replacing a sample's library by UNKNOWN during training")
    parser.add_argument("--exclude_cell_types", type=str, default=None,
                    help="Ablation (N-1 type): comma list of cell types removed from the TRAIN and VALID "
                         "partitions after the fixed split (test unchanged, so the benchmark scores them). "
                         "Applies to both modalities.")
    parser.add_argument("--holdout_donors", type=str, default=None,
                    help="Ablation (donor hold-out): comma list of <modality>:<donor>, e.g. 'nuclei:5'; those "
                         "observations leave TRAIN and VALID after the fixed split (test unchanged). Donor = "
                         "prefix of meta_batch_id, as in data/split_manifest.tsv.")
    parser.add_argument("--run_name", type=str, default=None,
                    help="Use run/<run_name> as the run folder instead of run/run_<timestamp>_<pid>. Lets a "
                         "requeued SLURM job find its own folder and resume (--continue_training True).")

    args = parser.parse_args()
    if args.seed is not None:
        import torch as _torch
        random.seed(args.seed); np.random.seed(args.seed); _torch.manual_seed(args.seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(args.seed)
        print(f"Seeded random/numpy/torch with {args.seed}")
    # cellsfile = '/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/ABC/results/10X_cells_v2_v3_AIBS_g10k/integrated.h5ad'
    # nucleifile = '/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/ABC/results/10X_nuclei_v2_v3_AIBS_g10k/integrated.h5ad'

    
    
    if args.test_nuclei_model_on_cell_data:
        save_path = os.path.join(args.test_model_run,"test_on_cell_data")
        os.makedirs(save_path, exist_ok=True)
        ConditionalVAE.test_nuclei_model_with_cell_data(save_path,args)
        exit()

    
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pid = os.getpid()
    folder_name = args.run_name if getattr(args, "run_name", None) else "run_{}_{}".format(now_str, pid)
    run_dpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", folder_name)
    print(run_dpath)
    os.makedirs(run_dpath, exist_ok=True)
    # test model mode:
    if args.test_model_on_whole_dataset:
        ConditionalVAE.test_model_on_whole_dataset(run_dpath,args)
        exit()

    if args.test_model:
        ConditionalVAE.test_model_ABC(run_dpath,args)
        exit()

    if args.test_combined:
        ConditionalVAE.test_model_ABC_combined(run_dpath,args)
        exit()

    if args.test_combined_mixed_modality:
        ConditionalVAE.test_model_ABC_combined_mixed_modality(run_dpath,args)
        exit()

    if args.continue_training:
        if not args.continue_training_run_path and getattr(args, "run_name", None):
            args.continue_training_run_path = run_dpath  # resume in the named folder
        run_dpath = args.continue_training_run_path
        args.start_epoch = args.continue_training_epoch




    print(run_dpath)
    if args.use_raw_data:
        adata = ConditionalVAE.load_raw_data(args.data_path, args)
        
    else:
        integrated_dp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ABC_results",
        args.data_path,
        "integrated.h5ad")
        adata = ad.read_h5ad(integrated_dp)
        

    cells_header = adata.obs["cell_type"].tolist()
    gene_names = adata.var["gene_names"].tolist()
    print(f"Number of cells = {len(cells_header)}")
    cell_types = list(set([cell.split(".")[0] for cell in cells_header if cell.split(".")[0] != "doublet"]))
    
    print("unique cell types: ",set(cell_types))
    # print(cells_header[:100])
    chunk_rows = 1000 

    args.num_feature = len(gene_names)
    args.num_latent = np.power(2,args.latent_size)
    args.num_internal = parse_internal_size(args.internal_size)

    args_dict = vars(args)

    # Specify the filename to save the arguments
    filename = os.path.join(run_dpath,'arguments.json')

    # Save arguments to a JSON file
    with open(filename, 'w') as f:
        serializable_data = convert_to_serializable(args_dict)
        json.dump(serializable_data, f, indent=4)

    print(f"Arguments saved to {filename}")

    print("Do we have Log in the reconstruction loss? ",args.recons_with_log)
    print("Do we limite the L1 to first layer? ",args.limit_l1)
    print("Do use subset of data? ",args.subset)
    print("Do we are fine tuning the model? ",args.fine_tune_model)
    
    print(args_dict)

    if args.fine_tune_model:
        ConditionalVAE.fine_tune_model(adata, cell_types, run_dpath,args)
        exit()

    if args.run_debug_function:
        ConditionalVAE.debug_model(cell_types,run_dpath,args)
        exit()

    if args.fine_tune_stage3:
        ConditionalVAE.train_loop_ABC_stage3(adata, cell_types, run_dpath,args)
        exit()

    if args.fine_tune_cond_stage3:
        ConditionalVAE.train_cond_stage3(adata, cell_types, run_dpath, args)
        exit()

    if args.fine_tune_stage3_v2:
        ConditionalVAE.train_loop_ABC_stage3_v2(adata, cell_types, run_dpath,args)
        exit()
    
    if args.fine_tune_stage2_v2:
        ConditionalVAE.train_loop_ABC_stage2_v2(adata, cell_types, run_dpath,args)
        exit()

    model, loss_list = ConditionalVAE.train_loop_ABC(adata, cell_types, run_dpath,args)

    # if args.testCode == True:
    #     model, loss_list = ConditionalVAE.train_loop_test(cell_types, cellsfile, run_dpath, args)
    # if args.model == "vae":
    #     model, loss_list = ConditionalVAE.train_loop(cell_types, cellsfile, run_dpath,args)
    # if args.model == "vae_l1":
    #     model, loss_list = ConditionalVAE.train_loop_L1(cell_types, cellsfile, run_dpath,args)
    # elif args.model == "vae_sl":
    #     model, loss_list = ConditionalVAE.train_loop_SL(cell_types, cellsfile, run_dpath,args)
    

    #save loss to file
    # header = "epoch,loss,recons_loss,KL_div"
    loss_file = os.path.join(run_dpath,"loss.txt")

    print("save loss info")
    with open(loss_file, 'w') as file:
        # file.write(header + '\n')  
        for row in loss_list:
            line = ','.join(map(str, row))
            file.write(line + '\n')

    print("Finish. -----END-----")
    if args.alignment_weight is None:
        args.alignment_weight = args.alpha
    if args.anneal_alpha:
        args.anneal_alignment = True
    if args.anneal_alpha_start_epoch != 0:
        args.anneal_alignment_start_epoch = args.anneal_alpha_start_epoch
    if args.anneal_alpha_epoch != 20:
        args.anneal_alignment_epoch = args.anneal_alpha_epoch
    if args.fine_tune_stage3_v2:
        # Preserve old behavior where beta scaled the per-modality JSD terms.
        args.cell_jsd_weight *= args.beta
        args.nuclei_jsd_weight *= args.beta
        if args.kl_anneal:
            args.anneal_cell_jsd_weight = True
            args.anneal_nuclei_jsd_weight = True
            args.cell_jsd_weight_start_epoch = args.kl_anneal_start_epoch
            args.nuclei_jsd_weight_start_epoch = args.kl_anneal_start_epoch
            args.cell_jsd_weight_anneal_epoch = args.kl_anneal_epoch
            args.nuclei_jsd_weight_anneal_epoch = args.kl_anneal_epoch
        # Stage-3 now applies all weighting in the joint loop.
        args.beta = 1.0
