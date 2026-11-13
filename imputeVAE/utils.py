import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def setup_device():
    """Setup and return the compute device. Call at start of training functions."""
    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        try:
            test_tensor = torch.randn(10, 10, device=device)
            del test_tensor
            torch.cuda.empty_cache()
            print("GPU test successful")
        except Exception as e:
            print(f"GPU test failed: {e}")
            print("Falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU")
    print(f"Final device: {device}")
    return device

def verify_gpu_usage(model, data, step_name=""):
    print(f"\n=== GPU Check: {step_name} ===")
    print(f"Model on GPU: {next(model.parameters()).is_cuda}")
    print(f"Data on GPU: {data.is_cuda if hasattr(data, 'is_cuda') else 'Not a tensor'}")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print("========================\n")

def save_latent_distribution_histograms(mu, log_var, loss,recons_loss, div, epoch, save_path):
    mu = mu.detach().cpu().numpy().flatten()
    log_var = log_var.detach().cpu().numpy().flatten()
    loss = loss.detach().cpu().numpy().flatten()
    recons_loss = recons_loss.detach().cpu().numpy().flatten()
    div = div.detach().cpu().numpy().flatten()
    fig_save_dir = os.path.join(save_path,"latent_distribution_histograms")
    os.makedirs(fig_save_dir,exist_ok=True)
    # Plot mu
    plt.figure()
    plt.hist(mu, bins=50, color='skyblue')
    plt.title(f"mu distribution at epoch {epoch}")
    plt.xlabel("mu")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f"mu_hist_epoch_{epoch}.png"))
    plt.close()

    # Plot log_var
    plt.figure()
    plt.hist(log_var, bins=50, color='salmon')
    plt.title(f"log_var distribution at epoch {epoch}")
    plt.xlabel("log_var")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f"logvar_hist_epoch_{epoch}.png"))
    plt.close()

    # Plot loss
    plt.figure()
    plt.hist(loss, bins=50, color='salmon')
    plt.title(f"loss distribution at epoch {epoch}")
    plt.xlabel("loss")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f"loss_hist_epoch_{epoch}.png"))
    plt.close()

     # Plot recons_loss
    plt.figure()
    plt.hist(recons_loss, bins=50, color='salmon')
    plt.title(f"recons_loss distribution at epoch {epoch}")
    plt.xlabel("recons_loss")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f"recons_loss_hist_epoch_{epoch}.png"))
    plt.close()

     # Plot div
    plt.figure()
    plt.hist(div, bins=50, color='salmon')
    plt.title(f"div distribution at epoch {epoch}")
    plt.xlabel("div")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f"div_hist_epoch_{epoch}.png"))
    plt.close()

def contains_nan(matrix):
    # Convert list to numpy array
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    
    # Check for numpy array
    if isinstance(matrix, np.ndarray):
        return np.isnan(matrix).any()
    
    # Check for torch tensor
    elif isinstance(matrix, torch.Tensor):
        return torch.isnan(matrix).any().item()
    
    else:
        raise TypeError(f"Unsupported data type: {type(matrix)}")


def save_epoch_save_dict(valid_epoch_save_dict, save_path, epoch, handle="train", save_file_path=None):
    if save_file_path is None:
        save_file_path = os.path.join(save_path, "epoch_dump", f"epoch_{epoch}_{handle}.json")
    with open(save_file_path, 'w') as fp:
        json.dump(valid_epoch_save_dict, fp)
    print(f"Epoch {epoch} {handle} save dict saved to {save_file_path}")

def load_args_from_json(json_path):
    with open(json_path, 'r') as f:
        args_dict = json.load(f)
    return args_dict
