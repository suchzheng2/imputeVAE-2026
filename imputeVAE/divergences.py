import torch
import numpy as np

def wasserstein_distance_gaussian(mu1, log_var1, mu2, log_var2):
    """
    Compute approximate 2-Wasserstein distance between two multivariate Gaussians.
    
    For Gaussians N(μ₁, Σ₁) and N(μ₂, Σ₂), the exact formula is:
    W₂²(P,Q) = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁^(1/2) Σ₂ Σ₁^(1/2)))
    
    For diagonal covariances, this simplifies to:
    W₂²(P,Q) = ||μ₁ - μ₂||² + ||σ₁ - σ₂||²
    """
    # Convert log_var to standard deviations
    sigma1 = torch.exp(0.5 * log_var1)
    sigma2 = torch.exp(0.5 * log_var2)
    
    # Mean difference term: ||μ₁ - μ₂||²
    mu_diff = torch.sum((mu1 - mu2) ** 2, dim=1)
    
    # Covariance difference term: ||σ₁ - σ₂||²
    sigma_diff = torch.sum((sigma1 - sigma2) ** 2, dim=1)
    
    # Return mean across batch
    return torch.mean(mu_diff + sigma_diff)

def compute_mc_jsd_from_latents(mu, log_var, num_samples=1000):
    # Sample z_q ~ q(z|x)
    std = torch.exp(0.5 * log_var)
    eps_q = torch.randn((num_samples, mu.size(1)), device=mu.device)
    z_q = mu + std * eps_q

    # Sample z_p ~ p(z)
    z_p = torch.randn_like(z_q)

    # Define log q(z) - log of probability density
    def log_q(z, mu, logvar):
        const = torch.log(torch.tensor(2 * torch.pi, device=logvar.device, dtype=logvar.dtype))
        return -0.5 * (logvar + const) - 0.5 * ((z - mu) ** 2) / torch.exp(logvar)

    # Define log p(z)
    def log_p(z):
        const = torch.log(torch.tensor(2 * torch.pi, device=z.device, dtype=z.dtype))
        return -0.5 * const - 0.5 * z ** 2

    # Compute log q(z_q), log p(z_q), log p(z_p), log q(z_p)
    # logq_zq = log_q(z_q, mu, log_var).sum(dim=1)
    # logp_zq = log_p(z_q).sum(dim=1)
    # logp_zp = log_p(z_p).sum(dim=1)
    # logq_zp = log_q(z_p, mu, log_var).sum(dim=1)

    logq_zq = log_q(z_q, mu, log_var)
    logp_zq = log_p(z_q)
    logp_zp = log_p(z_p)
    logq_zp = log_q(z_p, mu, log_var)

    # Compute log m(z_q), log m(z_p) using log-sum-exp trick
    logm_zq = torch.logsumexp(torch.stack([logq_zq, logp_zq], dim=0), dim=0) - torch.log(torch.tensor(2.0, device=mu.device))
    logm_zp = torch.logsumexp(torch.stack([logq_zp, logp_zp], dim=0), dim=0) - torch.log(torch.tensor(2.0, device=mu.device))

    # Compute expectations
    jsd_q = 0.5 * torch.mean(logq_zq - logm_zq)
    jsd_p = 0.5 * torch.mean(logp_zp - logm_zp)

    jsd_total = jsd_q + jsd_p

    return jsd_total

def compute_mc_jsd_from_latents_2(nc_mu, nc_log_var, c_mu, c_log_var, num_samples=1000):
    # Sample z_q ~ q(z|x)
    std = torch.exp(0.5 * nc_log_var)
    eps_q = torch.randn((num_samples, nc_mu.size(1)), device=nc_mu.device)
    z_q = nc_mu + std * eps_q

    # Sample z_p ~ p(z)
    std = torch.exp(0.5 * c_log_var)
    eps_p = torch.randn((num_samples, c_mu.size(1)), device=c_mu.device)
    z_p = c_mu + std * eps_p

    # Define log q(z) - log of probability density
    def log_q(z, mu, logvar):
        const = torch.log(torch.tensor(2 * torch.pi, device=logvar.device, dtype=logvar.dtype))
        return -0.5 * (logvar + const) - 0.5 * ((z - mu) ** 2) / torch.exp(logvar)

    # Define log p(z)
    def log_p(z):
        const = torch.log(torch.tensor(2 * torch.pi, device=z.device, dtype=z.dtype))
        return -0.5 * const - 0.5 * z ** 2

    # Compute log q(z_q), log p(z_q), log p(z_p), log q(z_p)
    logq_zq = log_q(z_q, nc_mu, nc_log_var).sum(dim=1)
    logp_zq = log_p(z_q).sum(dim=1)
    logp_zp = log_p(z_p).sum(dim=1)
    logq_zp = log_q(z_p, c_mu, c_log_var).sum(dim=1)

    # Compute log m(z_q), log m(z_p) using log-sum-exp trick
    logm_zq = torch.logsumexp(torch.stack([logq_zq, logp_zq], dim=0), dim=0) - torch.log(torch.tensor(2.0, device=nc_mu.device))
    logm_zp = torch.logsumexp(torch.stack([logq_zp, logp_zp], dim=0), dim=0) - torch.log(torch.tensor(2.0, device=c_mu.device))

    # Compute expectations
    jsd_q = 0.5 * torch.mean(logq_zq - logm_zq)
    jsd_p = 0.5 * torch.mean(logp_zp - logm_zp)

    jsd_total = jsd_q + jsd_p

    return jsd_total

def compute_mc_jsd_from_two_latents_flexible(mu1, log_var1, mu2, log_var2, 
                                           num_samples1=None, num_samples2=None, 
                                           default_samples=1000):
    """
    Compute Monte Carlo Jensen-Shannon divergence with flexible sample sizes.
    
    Args:
        mu1, log_var1: Parameters of first distribution [batch_size1, latent_dim]
        mu2, log_var2: Parameters of second distribution [batch_size2, latent_dim]  
        num_samples1: Number of samples from first distribution (optional)
        num_samples2: Number of samples from second distribution (optional)
        default_samples: Default number of samples if not specified
    
    Returns:
        Jensen-Shannon divergence between the two distributions
    """
    
    # Handle different batch sizes - use the maximum or specified values
    if num_samples1 is None:
        num_samples1 = mu1.size(0)  # Use actual batch size
    if num_samples2 is None:
        num_samples2 = mu2.size(0)  # Use actual batch size
    
    # Sample z1 ~ q1(z) (first distribution)
    std1 = torch.exp(0.5 * log_var1)
    # Create sampling indices to handle different batch sizes
    batch_idx1 = torch.randint(0, mu1.size(0), (num_samples1,), device=mu1.device)
    mu1_sampled = mu1[batch_idx1]
    std1_sampled = std1[batch_idx1]
    
    eps1 = torch.randn((num_samples1, mu1.size(1)), device=mu1.device)
    z1 = mu1_sampled + std1_sampled * eps1

    # Sample z2 ~ q2(z) (second distribution)  
    std2 = torch.exp(0.5 * log_var2)
    batch_idx2 = torch.randint(0, mu2.size(0), (num_samples2,), device=mu2.device)
    mu2_sampled = mu2[batch_idx2]
    std2_sampled = std2[batch_idx2]
    
    eps2 = torch.randn((num_samples2, mu2.size(1)), device=mu2.device)
    z2 = mu2_sampled + std2_sampled * eps2

    # Log probability for a Gaussian distribution
    def log_gaussian_vectorized(z, mu, logvar):
        """Compute log probability handling broadcasting between samples and batch"""
        # z: [num_samples, latent_dim]
        # mu, logvar: [batch_size, latent_dim]
        
        # Expand for broadcasting: z[samples, 1, dim], mu[1, batch, dim]
        z_exp = z.unsqueeze(1)  # [num_samples, 1, latent_dim]
        mu_exp = mu.unsqueeze(0)  # [1, batch_size, latent_dim]
        logvar_exp = logvar.unsqueeze(0)  # [1, batch_size, latent_dim]
        
        const = torch.log(torch.tensor(2 * torch.pi, device=logvar.device, dtype=logvar.dtype))
        log_prob = -0.5 * (logvar_exp + const + ((z_exp - mu_exp) ** 2) / torch.exp(logvar_exp))
        return log_prob.sum(dim=2)  # Sum over latent dimensions: [num_samples, batch_size]

    # Compute log probabilities for all combinations
    # For samples from q1 evaluated under both distributions
    log_q1_z1 = log_gaussian_vectorized(z1, mu1, log_var1)  # [num_samples1, batch_size1]
    log_q2_z1 = log_gaussian_vectorized(z1, mu2, log_var2)  # [num_samples1, batch_size2]
    
    # For samples from q2 evaluated under both distributions
    log_q1_z2 = log_gaussian_vectorized(z2, mu1, log_var1)  # [num_samples2, batch_size1]
    log_q2_z2 = log_gaussian_vectorized(z2, mu2, log_var2)  # [num_samples2, batch_size2]

    # Convert to mixture distributions by averaging over batches
    # This treats each batch as a mixture distribution
    log_q1_z1_mix = torch.logsumexp(log_q1_z1, dim=1) - torch.log(torch.tensor(mu1.size(0), dtype=torch.float, device=mu1.device))
    log_q2_z1_mix = torch.logsumexp(log_q2_z1, dim=1) - torch.log(torch.tensor(mu2.size(0), dtype=torch.float, device=mu2.device))
    
    log_q1_z2_mix = torch.logsumexp(log_q1_z2, dim=1) - torch.log(torch.tensor(mu1.size(0), dtype=torch.float, device=mu1.device))
    log_q2_z2_mix = torch.logsumexp(log_q2_z2, dim=1) - torch.log(torch.tensor(mu2.size(0), dtype=torch.float, device=mu2.device))

    # Compute log m(z) = log(0.5 * q1(z) + 0.5 * q2(z))
    log_m_z1 = torch.logsumexp(torch.stack([log_q1_z1_mix, log_q2_z1_mix], dim=0), dim=0) - torch.log(torch.tensor(2.0, device=mu1.device))
    log_m_z2 = torch.logsumexp(torch.stack([log_q1_z2_mix, log_q2_z2_mix], dim=0), dim=0) - torch.log(torch.tensor(2.0, device=mu1.device))

    # Compute JSD components with proper weighting for different sample sizes
    jsd_1 = 0.5 * torch.mean(log_q1_z1_mix - log_m_z1)
    jsd_2 = 0.5 * torch.mean(log_q2_z2_mix - log_m_z2)

    return jsd_1 + jsd_2

def compute_mc_jsd_simple_different_sizes(mu1, log_var1, mu2, log_var2, 
                                        num_samples1=1000, num_samples2=1000):
    """
    Simpler version: sample fixed numbers from empirical mixture distributions.
    This is often more stable and easier to understand.
    """
    
    # Sample from empirical mixture of first distribution
    batch_idx1 = torch.randint(0, mu1.size(0), (num_samples1,), device=mu1.device)
    mu1_samples = mu1[batch_idx1]
    log_var1_samples = log_var1[batch_idx1]
    std1_samples = torch.exp(0.5 * log_var1_samples)
    eps1 = torch.randn_like(std1_samples)
    z1 = mu1_samples + std1_samples * eps1
    
    # Sample from empirical mixture of second distribution  
    batch_idx2 = torch.randint(0, mu2.size(0), (num_samples2,), device=mu2.device)
    mu2_samples = mu2[batch_idx2]
    log_var2_samples = log_var2[batch_idx2]
    std2_samples = torch.exp(0.5 * log_var2_samples)
    eps2 = torch.randn_like(std2_samples)
    z2 = mu2_samples + std2_samples * eps2

    # Now use standard JSD calculation with these empirical samples
    # Create "pseudo-batches" by averaging parameters
    mu1_avg = mu1.mean(dim=0, keepdim=True)
    log_var1_avg = log_var1.mean(dim=0, keepdim=True) 
    mu2_avg = mu2.mean(dim=0, keepdim=True)
    log_var2_avg = log_var2.mean(dim=0, keepdim=True)

    # Log probability for a Gaussian distribution
    def log_gaussian(z, mu, logvar):
        const = torch.log(torch.tensor(2 * torch.pi, device=logvar.device, dtype=logvar.dtype))
        return -0.5 * (logvar + const + ((z - mu) ** 2) / torch.exp(logvar))

    # Compute log probabilities using averaged parameters
    log_q1_z1 = log_gaussian(z1, mu1_avg, log_var1_avg).sum(dim=1)
    log_q2_z1 = log_gaussian(z1, mu2_avg, log_var2_avg).sum(dim=1)
    
    log_q1_z2 = log_gaussian(z2, mu1_avg, log_var1_avg).sum(dim=1)
    log_q2_z2 = log_gaussian(z2, mu2_avg, log_var2_avg).sum(dim=1)

    # Compute mixture probabilities
    log_m_z1 = torch.logsumexp(torch.stack([log_q1_z1, log_q2_z1], dim=0), dim=0) - torch.log(torch.tensor(2.0, device=mu1_avg.device))
    log_m_z2 = torch.logsumexp(torch.stack([log_q1_z2, log_q2_z2], dim=0), dim=0) - torch.log(torch.tensor(2.0, device=mu1_avg.device))

    # Compute JSD components
    jsd_1 = 0.5 * torch.mean(log_q1_z1 - log_m_z1)
    jsd_2 = 0.5 * torch.mean(log_q2_z2 - log_m_z2)

    return jsd_1 + jsd_2

def compute_mc_jsd_from_two_latents(mu1, log_var1, mu2, log_var2):
    # Sample z1 ~ q1(z) (first distribution)
    std1 = torch.exp(0.5 * log_var1)
    eps1 = torch.randn((mu1.size(0), mu1.size(1)), device=mu1.device)
    z1 = mu1 + std1 * eps1

    # Sample z2 ~ q2(z) (second distribution)
    std2 = torch.exp(0.5 * log_var2)
    eps2 = torch.randn((mu2.size(0), mu2.size(1)), device=mu2.device)
    z2 = mu2 + std2 * eps2

    # Log probability for a Gaussian distribution
    def log_gaussian(z, mu, logvar):
        const = torch.log(torch.tensor(2 * torch.pi, device=logvar.device, dtype=logvar.dtype))
        return -0.5 * (logvar + const + ((z - mu) ** 2) / torch.exp(logvar))

    # Compute log probabilities
    # For samples from q1:
    log_q1_z1 = log_gaussian(z1, mu1, log_var1)
    log_q2_z1 = log_gaussian(z1, mu2, log_var2)
    
    # For samples from q2:
    log_q1_z2 = log_gaussian(z2, mu1, log_var1)
    log_q2_z2 = log_gaussian(z2, mu2, log_var2)

    # Compute log m(z) = log(0.5 * q1(z) + 0.5 * q2(z))
    log_m_z1 = torch.logsumexp(torch.stack([log_q1_z1, log_q2_z1], dim=0), dim=0) - torch.log(torch.tensor(2.0, device=mu1.device))

    log_m_z2 = torch.logsumexp(torch.stack([log_q1_z2, log_q2_z2], dim=0), dim=0) - torch.log(torch.tensor(2.0, device=mu1.device))

    # Compute JSD components
    jsd_1 = 0.5 * torch.mean(log_q1_z1 - log_m_z1)
    jsd_2 = 0.5 * torch.mean(log_q2_z2 - log_m_z2)

    return jsd_1 + jsd_2

def compute_per_sample_jsd(mu, log_var, num_samples=50):
    batch_size, latent_dim = mu.shape
    std = torch.exp(0.5 * log_var)

    # [batch, num_samples, latent_dim]
    eps_q = torch.randn(batch_size, num_samples, latent_dim, device=mu.device)
    z_q = mu.unsqueeze(1) + std.unsqueeze(1) * eps_q  # q(z|x) samples

    # Prior samples: shape [num_samples, latent_dim], broadcast to [batch, num_samples, latent_dim]
    z_p = torch.randn(num_samples, latent_dim, device=mu.device).unsqueeze(0).expand(batch_size, -1, -1)

    def log_normal(z, mu, logvar):
        const = torch.log(torch.tensor(2 * torch.pi, device=z.device, dtype=z.dtype))
        return -0.5 * (logvar.unsqueeze(1) + const) - 0.5 * ((z - mu.unsqueeze(1)) ** 2) / torch.exp(logvar.unsqueeze(1))

    log_q_zq = log_normal(z_q, mu, log_var).sum(dim=2)      # [batch, num_samples]
    log_p_zq = (-0.5 * (z_q ** 2 + torch.log(torch.tensor(2 * torch.pi, device=mu.device)))).sum(dim=2)

    log_q_zp = log_normal(z_p, mu, log_var).sum(dim=2)      # [batch, num_samples]
    log_p_zp = (-0.5 * (z_p ** 2 + torch.log(torch.tensor(2 * torch.pi, device=mu.device)))).sum(dim=2)

    logm_zq = torch.logsumexp(torch.stack([log_q_zq, log_p_zq], dim=2), dim=2) - np.log(2)
    logm_zp = torch.logsumexp(torch.stack([log_q_zp, log_p_zp], dim=2), dim=2) - np.log(2)

    jsd_q = 0.5 * (log_q_zq - logm_zq).mean(dim=1)  # [batch]
    jsd_p = 0.5 * (log_p_zp - logm_zp).mean(dim=1)  # [batch]

    jsd = jsd_q + jsd_p  # [batch]
    return jsd.mean()    # scalar


