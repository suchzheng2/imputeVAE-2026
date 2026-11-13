import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .divergences import compute_mc_jsd_from_latents, compute_mc_jsd_from_two_latents, \
    compute_mc_jsd_from_two_latents_flexible, wasserstein_distance_gaussian
from .utils import device, contains_nan

class VAE(nn.Module):
    def __init__(self, input_size=20000, num_internal=128, num_latent=128):
        super().__init__()

        # Allow single int or list
        if isinstance(num_internal, int):
            num_internal = [num_internal]

        self.input_size = input_size
        self.num_internal = num_internal
        self.num_latent = num_latent

        # ----- Build encoder -----
        encoder_layers = []
        in_dim = input_size
        for h in self.num_internal:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.LeakyReLU(0.1))
            in_dim = h

        # Final layer into latent
        encoder_layers.append(nn.Linear(self.num_internal[-1], self.num_latent))
        encoder_layers.append(nn.LeakyReLU(0.1))

        self.encoder = nn.Sequential(*encoder_layers)


        # ----- Build decoder -----
        decoder_layers = []
        in_dim = self.num_latent
        for h in reversed(self.num_internal):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.LeakyReLU(0.1))
            in_dim = h

        decoder_layers.append(nn.Linear(self.num_internal[0], input_size))
        decoder_layers.append(nn.LeakyReLU(0.1))

        self.decoder = nn.Sequential(*decoder_layers)

        self.mu = nn.Linear(self.num_latent, self.num_latent)
        self.log_var = nn.Linear(self.num_latent, self.num_latent)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var*0.5)
        
        print("--------------------------------")
        print("Debug:")
        print(f"veofre clamp , std: {std[:10]}") 
        
        std = torch.clamp(std, min=1e-6)  # Prevent zero std
        eps = torch.randn_like(std)

       
        print("--------------------------------")
        print(f"mu: {mu[:10]}")
        print(f"log_var: {log_var[:10]}")
        print(f"std: {std[:10]}")
        print(f"eps: {eps[:10]}")
        print(f"mu + eps*std: {mu[:10] + eps[:10]*std[:10]}")
        print(f"mu + eps*std first 10 elements: {mu[:10] + eps[:10]*std[:10][:10]}")
        print("--------------------------------")
        return mu + eps*std

    def forward(self,x):
        # print("In forward")
        # print("x:",x)
        encoded = self.encoder(x)
        # print("encoded:",encoded)
        mu = self.mu(encoded)
        # print("mu:",mu)
        log_var = self.log_var(encoded)
        print("--------------------------------")
        print("Debug:")
        print(f"log_var before clamp: {log_var[:10]}") 
        log_var = torch.clamp(log_var, min=-20, max=20)
        print(f"log_var after clamp: {log_var[:10]}") 
        print("--------------------------------")
        # print("log_var:",log_var)
        z = self.reparameterize(mu, log_var)
        # print("z:",z)
        decoded = self.decoder(z)
        # print("decoded:",decoded)
        return encoded, decoded, mu, log_var

    def sample(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.num_latent).to(next(self.parameters()).device)
            samples = self.decoder(z)
        return samples

    def loss_function(self,recon_x, x, mu, log_var, args):

        # print("loss function input: ")
        # print("x contains NaN? ",contains_nan(x))
        # print("recon_x contains NaN? ",contains_nan(recon_x))
        # print("mu: ",mu)
        # print("log_var: ",log_var)

        if args.add_mu_clamp:
            mu = torch.clamp(mu, max = args.add_mu_clamp, min = -5)
            # print("mu after clamp: ",mu)
        
        if args.add_log_var_clamp:
            log_var = torch.clamp(log_var, max = args.add_log_var_clamp, min = -5)
            # print("log_var after clamp: ",log_var)

        rloss = nn.MSELoss()

        #deal with old arguments
        if args.recons_with_log:
            args.recons_with_log_sqrt = True
        else:
            args.recons_with_sqrt = True


        #Loss function options:
        if args.recons_loss_type == 0:
            reconstruction_loss = rloss(recon_x,x)
        elif args.recons_loss_type == 1:
            reconstruction_loss = torch.sqrt(rloss(recon_x,x))
        elif args.recons_loss_type == 2:
            reconstruction_loss = torch.sqrt(rloss(torch.log(recon_x + 1),torch.log(x + 1)))
        else:
            #if not indicated, default as MSE
            reconstruction_loss = rloss(recon_x,x)
        if getattr(args, 'clamp_recons_loss', True):
            reconstruction_loss = torch.clamp(reconstruction_loss, min=1e-6, max=1-1e-6)
        KL_div = -0.5*torch.sum(1+log_var - mu.pow(2) - log_var.exp())
    
        return reconstruction_loss + args.beta*KL_div,reconstruction_loss,KL_div


class VAE_L1(VAE): 
    def __init__(self, *args, l1_coef = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1_coefficient = l1_coef
    
    def loss_function(self, recon_x, x, mu, log_var, args):
        _, reconstruction_loss, KL_div = super().loss_function(recon_x, x, mu, log_var, args)[:3]

        L1_reg = torch.tensor(0., requires_grad=True)
        if args.limit_l1:
            # print(next(self.encoder.children()))
            first_linear_layer = next(self.encoder.children())
            L1_reg = torch.norm(first_linear_layer.weight, 1)
            # print(L1_reg)
        elif args.avg_l1:
            total_params = 0
            for name, param in self.named_parameters():
                if 'weight' in name:
                    L1_reg = L1_reg + torch.norm(param, 1)
                    total_params += param.numel()
            if total_params > 0:
                L1_reg /= total_params  # Normalize the L1 regularization term

        else:
            for name, param in self.named_parameters():
                if 'weight' in name:
                    L1_reg = L1_reg + torch.norm(param, 1)

        # print("before return:")
        # print("reconstruction_loss:",reconstruction_loss)
        if args.norm_KL:
            KL_div = KL_div / recon_x.shape[0] 

        if args.kl_clamp:
            KL_div = torch.clamp(KL_div, min=1.0)
        
        loss =  reconstruction_loss + args.beta * KL_div + self.l1_coefficient * L1_reg

        if getattr(args, "use_soft_constraint", False):
            mu_penalty = args.soft_constraint_lambda * torch.mean(mu**2)
            logvar_penalty = args.soft_constraint_lambda * torch.mean(log_var**2)
            loss += mu_penalty + logvar_penalty

        
        return loss, reconstruction_loss, KL_div, L1_reg

class VAE_JSD(VAE):
    LN2 = np.log(2.0)  # theoretical max of JSD

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_function(self, recon_x, x, mu, log_var, args):
        _, reconstruction_loss, KL_div = super().loss_function(recon_x, x, mu, log_var, args)[:3]
        # JS_div = self.gjs_normal_loss(mu, log_var, args.JS_dual, args.JS_a, args.JS_invert_alpha)
        # JS_div = self.compute_classic_jsd_from_latents(mu, log_var)
        if args.debug:
            print("Debug:")
            print("--------------------------------")
            print(f"recon_x shape: {recon_x.shape}")
            print(f"x shape: {x.shape}")
            print(f"mu shape: {mu.shape}")
            print(f"log_var shape: {log_var.shape}")
            print(f"recon_x first 10 elements: {recon_x[:10]}")
            print(f"x first 10 elements: {x[:10]}")
            print(f"mu first 10 elements: {mu[:10]}")
            print(f"log_var first 10 elements: {log_var[:10]}")
            print("--------------------------------")
        JS_div = torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
        if args.beta > 0:
            JS_div = compute_mc_jsd_from_latents(mu, log_var, num_samples = mu.size(0))
            # JS_div = compute_per_sample_jsd(mu, log_var, num_samples = 100)
            if args.JS_norm:
                JS_div = JS_div/(JS_div+1)

        jsd_weight = 1.0 if getattr(args, "fine_tune_stage3_v2", False) else args.beta

        # Normalized loss mode: put recon and JSD on comparable [0,1] scale
        recon_loss_norm = getattr(args, "recon_loss_norm", 0.0)
        normalize = getattr(args, "normalize_losses", False) and recon_loss_norm > 0
        if normalize:
            scaled_recon = reconstruction_loss / recon_loss_norm
            scaled_jsd = JS_div / self.LN2
            loss = scaled_recon + jsd_weight * scaled_jsd
        else:
            loss = reconstruction_loss + jsd_weight * JS_div

        if getattr(args, "use_soft_constraint", False):
            mu_penalty = args.soft_constraint_lambda * torch.mean(mu ** 2)
            logvar_penalty = args.soft_constraint_lambda * torch.mean(log_var ** 2)
            loss = loss + mu_penalty + logvar_penalty
        return loss, reconstruction_loss, JS_div

    def fine_tune_loss_function(self, recon_x, x, nc_mu, nc_log_var, c_mu, c_log_var, args):
        if args.debug:
            print("--------------------------------")
            print("Debug:")
            print(f"nc_log_var shape: {nc_log_var.shape}")
            print(f"c_mu shape: {c_mu.shape}")
            print(f"c_log_var shape: {c_log_var.shape}")
            print(f"nc_mu mean: {nc_mu.mean():.6f}, std: {nc_mu.std():.6f}")
            print(f"nc_log_var mean: {nc_log_var.mean():.6f}, std: {nc_log_var.std():.6f}")
            print(f"c_mu mean: {c_mu.mean():.6f}, std: {c_mu.std():.6f}")
            print(f"c_log_var mean: {c_log_var.mean():.6f}, std: {c_log_var.std():.6f}")
            print("--------------------------------")
        _, reconstruction_loss, KL_div = super().loss_function(recon_x, x, nc_mu, nc_log_var, args)[:3]
        JS_div = torch.tensor(0.0, device=nc_mu.device, dtype=nc_mu.dtype)
        nuclei_JS_div = torch.tensor(0.0, device=nc_mu.device, dtype=nc_mu.dtype)
        nuclei_cell_type_JS_div = torch.tensor(0.0, device=nc_mu.device, dtype=nc_mu.dtype)
        nuclei_cell_type_distance = torch.tensor(0.0, device=nc_mu.device, dtype=nc_mu.dtype)
        if args.beta > 0:
            nuclei_JS_div = compute_mc_jsd_from_latents(nc_mu, nc_log_var, num_samples = nc_mu.size(0))
            # JS_div = compute_per_sample_jsd(mu, log_var, num_samples = 100)
            if args.JS_norm:
                nuclei_JS_div = nuclei_JS_div/(nuclei_JS_div+1)

        if args.alpha > 0:
            # In your evaluation loop, add:
            if args.debug:
                print("Debug:")
                print(f"Cell mu mean: {c_mu.mean():.6f}, std: {c_mu.std():.6f}")
                print(f"Cell logvar mean: {c_log_var.mean():.6f}, std: {c_log_var.std():.6f}")
                print(f"Nuclei mu mean: {nc_mu.mean():.6f}, std: {nc_mu.std():.6f}")
                print(f"Nuclei logvar mean: {nc_log_var.mean():.6f}, std: {nc_log_var.std():.6f}")
            if args.ft_metrics == "jsd":
                if args.use_all_cell_mu_log_var:
                    nuclei_cell_type_distance = compute_mc_jsd_from_two_latents_flexible(nc_mu, nc_log_var, c_mu, c_log_var)
                else:   
                    nuclei_cell_type_distance = compute_mc_jsd_from_two_latents(nc_mu, nc_log_var, c_mu, c_log_var)
            elif args.ft_metrics == "wasserstein":
                nuclei_cell_type_distance = wasserstein_distance_gaussian(nc_mu, nc_log_var, c_mu, c_log_var)
            if args.JS_norm:
                nuclei_cell_type_distance = nuclei_cell_type_distance/(nuclei_cell_type_distance+1)
        loss =  reconstruction_loss + args.beta * nuclei_JS_div + args.alpha * nuclei_cell_type_distance
        if args.debug:
            print("--------------------------------")
            print("Debug:")
            # check shape of each loss item
            print(f"reconstruction_loss shape: {reconstruction_loss.shape}")
            print(f"nuclei_JS_div shape: {nuclei_JS_div.shape}")
            print(f"nuclei_cell_type_distance shape: {nuclei_cell_type_distance.shape}")
            print(f"loss shape: {loss.shape}")
            print(f"nc_mu shape: {nc_mu.shape}")
            print("--------------------------------")
        return loss, reconstruction_loss, nuclei_JS_div, nuclei_cell_type_distance   

class VAE_KL_Sparsity(VAE):  
    def __init__(self, *args, spar_coef = 0.05, rho = 0.01,  **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_coefficient = spar_coef  # Sparsity coefficient
        self.rho = rho  # Target sparsity level

    def loss_function(self, recon_x, x, mu, log_var, args):
        _, reconstruction_loss, KL_div = super().loss_function(recon_x, x, mu, log_var, args)[:3]
        sparsity_loss = self.sparse_coefficient * torch.sum(self.rho * torch.log(self.rho / torch.mean(mu, dim=0)) + 
                                                            (1 - self.rho) * torch.log((1 - self.rho) / (1 - torch.mean(mu, dim=0))))
        return reconstruction_loss + args.beta * KL_div + sparsity_loss, reconstruction_loss, KL_div, sparsity_loss


class VAE_NB(VAE):
    """
    VAE with Negative Binomial (NB) reconstruction loss for raw count data.
    Optionally supports Zero-Inflated Negative Binomial (ZINB).
    """
    def __init__(self, input_size=20000, num_internal=128, num_latent=128, use_zinb=False):
        super().__init__(input_size=input_size, num_internal=num_internal, num_latent=num_latent)
        self.use_zinb = use_zinb

        # Override decoder: remove final LeakyReLU so output is raw logits (rho)
        decoder_layers = []
        in_dim = self.num_latent
        for h in reversed(self.num_internal):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.LeakyReLU(0.1))
            in_dim = h
        # Final layer outputs logits (no activation)
        decoder_layers.append(nn.Linear(self.num_internal[0], input_size))
        self.decoder = nn.Sequential(*decoder_layers)

        # Global per-gene dispersion parameter (log scale, learned)
        self.log_theta = nn.Parameter(torch.zeros(input_size))

        # Optional ZINB dropout logits head (from last hidden layer)
        if self.use_zinb:
            self.zi_logits = nn.Linear(self.num_internal[0], input_size)

    def forward(self, x):
        print("In VAE_NB forward")
        # Encode
        encoded = self.encoder(x)
        mu_latent = self.mu(encoded)
        log_var_latent = self.log_var(encoded)
        log_var_latent = torch.clamp(log_var_latent, min=-20, max=20)

        # Reparameterize
        std = torch.exp(log_var_latent * 0.5)
        std = torch.clamp(std, min=1e-6)
        eps = torch.randn_like(std)
        z = mu_latent + eps * std

        # Decode to logits (rho)
        rho = self.decoder(z)

        # Compute library size per cell (sum of counts)
        library_size = x.sum(dim=1, keepdim=True)
        library_size = torch.clamp(library_size, min=1.0)

        # Convert logits to NB mean: mu_nb = library_size * softmax(rho)
        mu_nb = library_size * F.softmax(rho, dim=1)

        # Dispersion parameter (constrained positive)
        theta = F.softplus(self.log_theta)
        theta = torch.clamp(theta, min=1e-4, max=1e6)

        # Store NB params internally so loss_function can access them
        # without changing the shared interface
        self._nb_params = {"theta": theta}

        # Optional ZINB: compute dropout logits from the last hidden layer output
        if self.use_zinb:
            hidden = z
            for layer in list(self.decoder)[:-1]:
                hidden = layer(hidden)
            zi_logits = self.zi_logits(hidden)
            self._nb_params["zi_logits"] = zi_logits

        # Return same 4-tuple as base VAE: (encoded, decoded, mu, log_var)
        return encoded, mu_nb, mu_latent, log_var_latent

    def loss_function(self, recon_x, x, mu_latent, log_var_latent, args):
        """
        Same signature as base VAE. NB params are read from self._nb_params
        (set during forward()).
        """
        print("In VAE_NB loss_function")
        eps = 1e-8
        mu_nb = recon_x
        theta = self._nb_params["theta"]

        # NB log-likelihood per element
        # log NB(x | mu, theta) = lgamma(x+theta) - lgamma(theta) - lgamma(x+1)
        #                        + theta*log(theta/(theta+mu)) + x*log(mu/(theta+mu))
        log_theta_mu = torch.log(theta + mu_nb + eps)
        nb_log_prob = (
            torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
            + theta * (torch.log(theta + eps) - log_theta_mu)
            + x * (torch.log(mu_nb + eps) - log_theta_mu)
        )

        if self.use_zinb and "zi_logits" in self._nb_params:
            # ZINB loss
            pi = torch.sigmoid(self._nb_params["zi_logits"])
            nb_prob_zero = torch.exp(theta * (torch.log(theta + eps) - log_theta_mu))

            # x == 0 case: log(pi + (1-pi)*NB(0|mu,theta))
            zero_case = torch.log(pi + (1 - pi) * nb_prob_zero + eps)
            # x > 0 case: log(1-pi) + log NB(x|mu,theta)
            nonzero_case = torch.log(1 - pi + eps) + nb_log_prob

            # Select based on whether x is zero
            is_zero = (x < 0.5).float()
            reconstruction_loss = -torch.mean(is_zero * zero_case + (1 - is_zero) * nonzero_case)
        else:
            # Standard NB loss
            reconstruction_loss = -torch.mean(nb_log_prob)

        # KL divergence
        # KL_div = -0.5 * torch.sum(1 + log_var_latent - mu_latent.pow(2) - log_var_latent.exp())
        # JSD loss

        JS_div = torch.tensor(0.0, device=mu_latent.device, dtype=mu_latent.dtype)
        if args.beta > 0:
            JS_div = compute_mc_jsd_from_latents(mu_latent, log_var_latent, num_samples = mu_latent.size(0))
            # JS_div = compute_per_sample_jsd(mu, log_var, num_samples = 100)
            if args.JS_norm:
                JS_div = JS_div/(JS_div+1)

        jsd_weight = 1.0 if getattr(args, "fine_tune_stage3_v2", False) else args.beta
        loss =  reconstruction_loss + jsd_weight * JS_div
        return loss, reconstruction_loss, JS_div

    def cross_recons_loss(self, decoded_source, source_input, cross_target):
        """
        Compute NB reconstruction loss between decoded output from one modality
        and raw data from another modality, with library size re-scaling.

        Args:
            decoded_source: mu_nb from source modality forward pass
                            (library_size_source * softmax(rho))
            source_input: raw count input of the source modality (used to
                          recover gene proportions via its library size)
            cross_target: raw count data of the target modality to compare against

        Returns:
            Scalar NB negative log-likelihood loss.
        """
        eps = 1e-8
        theta = self._nb_params["theta"]

        # Recover gene proportions from decoded output
        source_library = source_input.sum(dim=1, keepdim=True).clamp(min=1.0)
        proportions = decoded_source / source_library  # softmax(rho)

        # Re-scale with cross-target library size
        cross_library = cross_target.sum(dim=1, keepdim=True).clamp(min=1.0)
        cross_mu_nb = cross_library * proportions

        # NB log-likelihood against cross-target counts
        log_theta_mu = torch.log(theta + cross_mu_nb + eps)
        nb_log_prob = (
            torch.lgamma(cross_target + theta)
            - torch.lgamma(theta)
            - torch.lgamma(cross_target + 1)
            + theta * (torch.log(theta + eps) - log_theta_mu)
            + cross_target * (torch.log(cross_mu_nb + eps) - log_theta_mu)
        )
        return -torch.mean(nb_log_prob)
