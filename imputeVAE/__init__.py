# imputeVAE - VAE-based cell/nucleus cytoplasm imputation
#
# This package provides backward compatibility so that:
#   import imputeVAE
#   imputeVAE.train_loop_ABC(...)
# works the same as the old:
#   import ConditionalVAE
#   ConditionalVAE.train_loop_ABC(...)

from .utils import (
    device, setup_device, verify_gpu_usage, save_latent_distribution_histograms,
    contains_nan, save_epoch_save_dict, load_args_from_json,
)
from .divergences import (
    wasserstein_distance_gaussian,
    compute_mc_jsd_from_latents,
    compute_mc_jsd_from_latents_2,
    compute_mc_jsd_from_two_latents_flexible,
    compute_mc_jsd_simple_different_sizes,
    compute_mc_jsd_from_two_latents,
    compute_per_sample_jsd,
)
from .models import VAE, VAE_L1, VAE_JSD, VAE_KL_Sparsity, VAE_NB
from .data import (
    myDataset, MyConcatDataset, myDataset_old,
    minmax_normalize_data, log_transform_data, quantile_normalize,
    load_and_normalize_data, load_raw_data, export_sample_manifest,
    balanced_split_by_cell_type, create_dataloaders, create_dataloaders_by_cell_type,
)
from .training import (
    train_loop_ABC, train_loop, train_loop_test, train_loop_L1, train_loop_SL,
)
from .staged_training import (
    train_loop_ABC_stage2_v2, train_loop_ABC_stage3, train_loop_ABC_stage3_v2,
)
from .cond_training import train_cond_stage3
from .models_cond import VAE_JSD_Cond, VAE_NB_Cond
from .testing import (
    test_model_on_whole_dataset, test_nuclei_model_with_cell_data,
    test_model_ABC, test_model_ABC_combined, load_mixed_dataset,
    test_model_ABC_combined_mixed_modality,
    evaluate_nuclei_model, combine_model, evaluate_combined_model, load_vae_jsd,
)
from .fine_tuning import fine_tune_model, debug_model
