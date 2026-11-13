import torch
import torch.nn as nn
from torch.nn import functional as F

from .models import VAE_JSD, VAE_NB


class VAE_NB_Cond(VAE_NB):
    """
    VAE with Negative Binomial loss and modality-conditioned decoder.

    Adds a learned modality embedding (0=cell, 1=nuclei) that is concatenated
    to the latent code z before the decoder.  The encoder is unchanged.

    When scalar_cond=True, uses a fixed scalar (1.0=cell, 0.0=nuclei) instead
    of a learned embedding — a traditional cVAE approach.
    """

    def __init__(self, input_size=20000, num_internal=128, num_latent=128,
                 embed_dim=8, use_zinb=False, scalar_cond=False):
        super().__init__(
            input_size=input_size,
            num_internal=num_internal,
            num_latent=num_latent,
            use_zinb=use_zinb,
        )
        self.scalar_cond = scalar_cond
        if scalar_cond:
            self.embed_dim = 1
            self.modality_emb = None
        else:
            self.embed_dim = embed_dim
            self.modality_emb = nn.Embedding(2, embed_dim)

        # Allow single int or list
        if isinstance(num_internal, int):
            num_internal = [num_internal]

        # Rebuild decoder with expanded input: num_latent + embed_dim
        decoder_layers = []
        in_dim = num_latent + self.embed_dim
        for h in reversed(num_internal):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.LeakyReLU(0.1))
            in_dim = h
        # Final layer outputs logits (no activation), same as VAE_NB
        decoder_layers.append(nn.Linear(num_internal[0], input_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def _get_modality_emb(self, batch_size, modality_id, device):
        """Return modality embedding [B, E] for the given integer id."""
        if self.scalar_cond:
            val = 1.0 if modality_id == 0 else 0.0
            return torch.full((batch_size, 1), val, dtype=torch.float32, device=device)
        ids = torch.full((batch_size,), modality_id, dtype=torch.long, device=device)
        return self.modality_emb(ids)

    def forward(self, x, modality_id=0):
        """
        Forward pass with modality conditioning.

        Args:
            x: [B, G] raw counts
            modality_id: int, 0=cell, 1=nuclei
        Returns:
            (encoded, mu_nb, mu_latent, log_var_latent)  — same 4-tuple as VAE_NB
        """
        # Encode (identical to VAE_NB)
        encoded = self.encoder(x)
        mu_latent = self.mu(encoded)
        log_var_latent = self.log_var(encoded)
        log_var_latent = torch.clamp(log_var_latent, min=-20, max=20)

        # Reparameterize
        std = torch.exp(log_var_latent * 0.5).clamp(min=1e-6)
        z = mu_latent + torch.randn_like(std) * std

        # Concatenate modality embedding
        emb = self._get_modality_emb(x.size(0), modality_id, x.device)
        z_cond = torch.cat([z, emb], dim=1)  # [B, L+E]

        # Decode to logits (rho)
        rho = self.decoder(z_cond)

        # NB mean
        library_size = x.sum(dim=1, keepdim=True).clamp(min=1.0)
        mu_nb = library_size * F.softmax(rho, dim=1)

        # Dispersion
        theta = F.softplus(self.log_theta).clamp(min=1e-4, max=1e6)
        self._nb_params = {"theta": theta}

        if self.use_zinb:
            hidden = z_cond
            for layer in list(self.decoder)[:-1]:
                hidden = layer(hidden)
            zi_logits = self.zi_logits(hidden)
            self._nb_params["zi_logits"] = zi_logits

        return encoded, mu_nb, mu_latent, log_var_latent

    def decode_cross(self, x_source, source_modality_id, target_modality_id):
        """
        Encode with source data, decode with target modality embedding.

        Used at eval time: e.g. encode nuclei (id=1), decode with cell
        embedding (id=0) to produce cell-like output.

        Returns mu_nb using source library size.
        """
        with torch.no_grad():
            encoded = self.encoder(x_source)
            mu_latent = self.mu(encoded)
            # Deterministic: use mu directly (no sampling)
            emb = self._get_modality_emb(x_source.size(0), target_modality_id,
                                         x_source.device)
            z_cond = torch.cat([mu_latent, emb], dim=1)
            rho = self.decoder(z_cond)
            library_size = x_source.sum(dim=1, keepdim=True).clamp(min=1.0)
            mu_nb = library_size * F.softmax(rho, dim=1)
        return mu_nb

    def decode_cross_train(self, mu_source, target_modality_id, target_library_size):
        """
        Cross-decode for training (with gradients).

        Takes pre-computed mu from source forward pass, decodes with target
        embedding and target library size. Gradients flow through decoder
        and modality embedding.

        Args:
            mu_source: [B, L] latent mu from source modality (detached from encoder)
            target_modality_id: int, modality to decode as
            target_library_size: [B, 1] library size of target data

        Returns:
            mu_nb: [B, G] NB mean (target_library * softmax(rho))
        """
        emb = self._get_modality_emb(mu_source.size(0), target_modality_id,
                                     mu_source.device)
        z_cond = torch.cat([mu_source.detach(), emb], dim=1)
        rho = self.decoder(z_cond)
        mu_nb = target_library_size * F.softmax(rho, dim=1)
        return mu_nb

    def load_pretrained_partial(self, pretrained_path, device=None):
        """
        Load weights from a pretrained VAE_NB model, expanding decoder
        layer 0 to accommodate the modality embedding dimensions.

        The embedding columns of decoder layer 0 are zero-initialized so
        the model initially behaves identically to the pretrained model
        regardless of modality_id.
        """
        if device is None:
            device = next(self.parameters()).device
        pretrained = torch.load(pretrained_path, map_location=device)

        model_dict = self.state_dict()

        # Separate matched and mismatched keys
        matched = {}
        mismatched = {}
        missing_in_pretrained = []
        for k, v in model_dict.items():
            if k in pretrained:
                if pretrained[k].shape == v.shape:
                    matched[k] = pretrained[k]
                else:
                    mismatched[k] = pretrained[k]
            else:
                missing_in_pretrained.append(k)

        # Load all matched weights
        model_dict.update(matched)

        # Handle decoder layer 0 weight expansion: [H, L] -> [H, L+E]
        dec_w_key = "decoder.0.weight"
        dec_b_key = "decoder.0.bias"
        if dec_w_key in mismatched:
            old_w = mismatched[dec_w_key]  # [H, L]
            new_w = model_dict[dec_w_key]  # [H, L+E]
            num_latent = old_w.shape[1]
            new_w[:, :num_latent] = old_w
            new_w[:, num_latent:] = 0.0
            model_dict[dec_w_key] = new_w
            print(f"  Decoder layer 0 weight expanded: {old_w.shape} -> {new_w.shape} "
                  f"(embedding cols zero-initialized)")

        if dec_b_key in pretrained and pretrained[dec_b_key].shape == model_dict[dec_b_key].shape:
            model_dict[dec_b_key] = pretrained[dec_b_key]

        self.load_state_dict(model_dict)

        print(f"  Loaded pretrained weights: {len(matched)} matched, "
              f"{len(mismatched)} expanded, {len(missing_in_pretrained)} new "
              f"({missing_in_pretrained})")


class VAE_JSD_Cond(VAE_JSD):
    """
    VAE with MSE/JSD loss and modality-conditioned decoder (and optionally encoder).

    Adds a modality signal that is concatenated to the latent code z before
    the decoder.  When cond_encoder=True, the signal is also concatenated to
    the input before the encoder, giving a full cVAE.

    Conditioning modes (controlled by scalar_cond):
      - scalar_cond=False (default): learned nn.Embedding(2, embed_dim)
      - scalar_cond=True: fixed scalar 1.0=cell / 0.0=nuclei, embed_dim forced to 1

    Library-size covariate (libsize_cond=True, added 2026-10-16):
      one observed scalar s = (log1p(total counts) - mean) / std is appended to
      the encoder input and to the decoder input, as a second conditioning
      scalar next to the modality signal (the two are never folded together).
      `mean`/`std` are buffers set from the training partition
      (`set_libsize_stats`) and travel with the checkpoint.  At cross-decode
      the target library is an explicit argument, so the output scale is a
      choice instead of a side effect of a latent dimension.  Default False
      reproduces the paper model exactly (same state_dict keys).

    Library (batch) covariate (library_cond=True, added 2026-11-06):
      a learned nn.Embedding(library_capacity + 1, library_dim) of the sequencing
      library (meta_batch_id, e.g. "4-c3A"; one vocabulary across both modalities,
      <run>/library_vocab.json, built from the training partition) is appended to
      the encoder input and to the decoder input next to the modality signal and
      the library-size scalar.  Index 0 is the UNKNOWN library: during training
      each sample's id is replaced by 0 with probability `library_dropout`, so
      the model learns a marginal ("average library") embedding that is used at
      inference for unseen libraries and as the default cross-decode target.
      The encoder sees the library so that the latent can be library-free; the
      decoder gets the target library explicitly (scVI-style conditioning).
      Default False keeps the state_dict identical to the earlier models.
    """

    def __init__(self, input_size=20000, num_internal=128, num_latent=128,
                 embed_dim=8, scalar_cond=False, cond_encoder=False,
                 libsize_cond=False, library_cond=False, library_capacity=32,
                 library_dim=4, library_dropout=0.0):
        super().__init__(
            input_size=input_size,
            num_internal=num_internal,
            num_latent=num_latent,
        )
        self.scalar_cond = scalar_cond
        self.cond_encoder = cond_encoder
        self.libsize_cond = libsize_cond
        self.libsize_dim = 1 if libsize_cond else 0
        self.library_cond = library_cond
        self.library_capacity = int(library_capacity)
        self.library_dim = int(library_dim) if library_cond else 0
        self.library_dropout = float(library_dropout)
        if library_cond:
            # row 0 = UNKNOWN library (trained through dropout); rows 1..capacity = vocabulary slots
            self.library_emb = nn.Embedding(self.library_capacity + 1, self.library_dim)
        else:
            self.library_emb = None
        if libsize_cond:
            # standardisation constants of log1p(library size), fixed from the training partition
            self.register_buffer("libsize_mean", torch.tensor(0.0))
            self.register_buffer("libsize_std", torch.tensor(1.0))
        if scalar_cond:
            self.embed_dim = 1
            self.modality_emb = None  # no learnable embedding
        else:
            self.embed_dim = embed_dim
            self.modality_emb = nn.Embedding(2, embed_dim)  # 0=cell, 1=nuclei

        # Allow single int or list
        if isinstance(num_internal, int):
            num_internal = [num_internal]

        # Rebuild encoder if conditioning it (modality signal and/or library size)
        if cond_encoder or libsize_cond or library_cond:
            encoder_layers = []
            in_dim = input_size + (self.embed_dim if cond_encoder else 0) + self.libsize_dim + self.library_dim
            for h in num_internal:
                encoder_layers.append(nn.Linear(in_dim, h))
                encoder_layers.append(nn.LeakyReLU(0.1))
                in_dim = h
            encoder_layers.append(nn.Linear(num_internal[-1], num_latent))
            encoder_layers.append(nn.LeakyReLU(0.1))
            self.encoder = nn.Sequential(*encoder_layers)

        # Rebuild decoder with expanded input: num_latent + embed_dim (+ 1 for library size)
        # Keep final LeakyReLU(0.1) to match base VAE/VAE_JSD
        decoder_layers = []
        in_dim = num_latent + self.embed_dim + self.libsize_dim + self.library_dim
        for h in reversed(num_internal):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.LeakyReLU(0.1))
            in_dim = h
        decoder_layers.append(nn.Linear(num_internal[0], input_size))
        decoder_layers.append(nn.LeakyReLU(0.1))
        self.decoder = nn.Sequential(*decoder_layers)

    # ------------------------------------------------------------------
    # Library-size covariate helpers
    # ------------------------------------------------------------------
    @staticmethod
    def log_library_size(x):
        """log1p(total raw counts) per sample, recovered from log1p-normalised input x.

        x is log1p(counts) over the training gene set, so expm1(x).sum() is the
        library size over exactly those genes (the quantity inference must use).
        Returns [B, 1].
        """
        return torch.log1p(torch.expm1(x).sum(dim=1, keepdim=True).clamp(min=0.0))

    def set_libsize_stats(self, mean, std):
        """Set the standardisation constants (training-partition log1p library size, both modalities pooled)."""
        if not self.libsize_cond:
            raise RuntimeError("set_libsize_stats called on a model without libsize_cond")
        if not std or float(std) <= 0:
            raise ValueError(f"libsize std must be positive, got {std}")
        self.libsize_mean.fill_(float(mean))
        self.libsize_std.fill_(float(std))

    def _standardize_libsize(self, log_lib):
        return (log_lib - self.libsize_mean) / self.libsize_std

    def _as_log_libsize_column(self, value, batch_size, like):
        """Accept a python float, 0-d / [B] / [B,1] tensor or array; return a [B, 1] tensor on `like`'s device."""
        t = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        if t.numel() == 1:
            return t.reshape(1, 1).expand(batch_size, 1)
        if t.numel() != batch_size:
            raise ValueError(f"target library size has {t.numel()} entries for a batch of {batch_size}")
        return t.reshape(batch_size, 1)

    def _encoder_input(self, x, emb, s, lemb=None):
        parts = [x]
        if self.cond_encoder:
            parts.append(emb)
        if self.libsize_cond:
            parts.append(s)
        if self.library_cond:
            parts.append(lemb)
        return torch.cat(parts, dim=1) if len(parts) > 1 else x

    def _decoder_input(self, z, emb, s, lemb=None):
        parts = [z, emb]
        if self.libsize_cond:
            parts.append(s)
        if self.library_cond:
            parts.append(lemb)
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # Library (batch) covariate helpers
    # ------------------------------------------------------------------
    def _as_library_ids(self, library_id, batch_size, device):
        """int / [B] array-or-tensor -> LongTensor [B]; None -> all UNKNOWN (0)."""
        if library_id is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        t = torch.as_tensor(library_id, dtype=torch.long, device=device)
        if t.numel() == 1:
            return t.reshape(1).expand(batch_size)
        if t.numel() != batch_size:
            raise ValueError(f"library_id has {t.numel()} entries for a batch of {batch_size}")
        return t.reshape(batch_size)

    def _get_library_emb(self, batch_size, device, library_id=None, library_emb=None, apply_dropout=False):
        """Library embedding [B, D]. `library_emb` ([D] or [B, D]) overrides the id lookup (e.g. a mean
        embedding over the cell libraries); `apply_dropout` replaces ids by UNKNOWN with p=library_dropout."""
        if not self.library_cond:
            return None
        if library_emb is not None:
            e = torch.as_tensor(library_emb, dtype=torch.float32, device=device)
            return e.reshape(1, -1).expand(batch_size, -1) if e.dim() == 1 else e.reshape(batch_size, -1)
        ids = self._as_library_ids(library_id, batch_size, device)
        if ids.max() > self.library_capacity:
            raise ValueError(f"library id {int(ids.max())} exceeds library_capacity {self.library_capacity}")
        if apply_dropout and self.library_dropout > 0:
            drop = torch.rand(batch_size, device=device) < self.library_dropout
            ids = torch.where(drop, torch.zeros_like(ids), ids)
        return self.library_emb(ids)

    def mean_library_emb(self, library_ids):
        """Mean embedding over a list of vocabulary ids (e.g. all cell libraries) -> [D] tensor."""
        ids = torch.as_tensor(list(library_ids), dtype=torch.long, device=self.library_emb.weight.device)
        return self.library_emb(ids).mean(dim=0).detach()

    def _get_modality_emb(self, batch_size, modality_id, device):
        """Return modality embedding [B, E] for the given integer id."""
        if self.scalar_cond:
            # Fixed scalar: 1.0 for cell (id=0), 0.0 for nuclei (id=1)
            val = 1.0 if modality_id == 0 else 0.0
            return torch.full((batch_size, 1), val, dtype=torch.float32, device=device)
        ids = torch.full((batch_size,), modality_id, dtype=torch.long, device=device)
        return self.modality_emb(ids)

    def forward(self, x, modality_id=0, log_libsize=None, library_id=None):
        """
        Forward pass with modality conditioning.

        Args:
            x: [B, G] log1p-normalized data
            modality_id: int, 0=cell, 1=nuclei
            log_libsize: optional [B] or [B,1] log1p library size; when None and
                libsize_cond is on, it is computed from x (the sample's own library).
            library_id: optional [B] vocabulary ids (library_cond only); None = UNKNOWN.
                Embedding dropout (-> UNKNOWN) is applied in training mode.
        Returns:
            (encoded, decoded, mu_latent, log_var_latent) — same 4-tuple as VAE_JSD
        """
        emb = self._get_modality_emb(x.size(0), modality_id, x.device)
        s = None
        if self.libsize_cond:
            log_lib = self.log_library_size(x) if log_libsize is None else \
                self._as_log_libsize_column(log_libsize, x.size(0), x)
            s = self._standardize_libsize(log_lib)
        lemb = self._get_library_emb(x.size(0), x.device, library_id=library_id, apply_dropout=self.training)

        # Encode — optionally with modality signal (and library size / library id) appended to input
        encoded = self.encoder(self._encoder_input(x, emb, s, lemb))
        mu_latent = self.mu(encoded)
        log_var_latent = self.log_var(encoded)
        log_var_latent = torch.clamp(log_var_latent, min=-20, max=20)

        # Reparameterize
        std = torch.exp(log_var_latent * 0.5).clamp(min=1e-6)
        z = mu_latent + torch.randn_like(std) * std

        # Decode with modality embedding (and the sample's own library size / library id)
        decoded = self.decoder(self._decoder_input(z, emb, s, lemb))

        return encoded, decoded, mu_latent, log_var_latent

    def encode_mu(self, x_source, source_modality_id, source_library_id=None):
        """Deterministic encoding: posterior mean mu [B, L] (no sampling, no grad)."""
        with torch.no_grad():
            source_emb = self._get_modality_emb(x_source.size(0), source_modality_id,
                                                x_source.device)
            s = self._standardize_libsize(self.log_library_size(x_source)) if self.libsize_cond else None
            lemb = self._get_library_emb(x_source.size(0), x_source.device, library_id=source_library_id)
            encoded = self.encoder(self._encoder_input(x_source, source_emb, s, lemb))
            return self.mu(encoded)

    def decode_cross(self, x_source, source_modality_id, target_modality_id,
                     target_log_libsize=None, source_library_id=None, target_library_id=None,
                     target_library_emb=None):
        """
        Encode with source data + source modality, decode with target modality.

        Used at eval time: e.g. encode nuclei (id=1), decode with cell
        embedding (id=0) to produce cell-like output.

        Args:
            target_log_libsize: only used when libsize_cond. log1p library size the
                output should be decoded at: a float (same for the whole batch, e.g.
                the per-cell-type median training-cell library) or a [B]/[B,1] tensor.
                None = the source sample's own library ("source" mode; reproduces the
                behaviour of the model without the covariate).
            source_library_id / target_library_id: library_cond only; vocabulary ids ([B] or
                int) of the source samples' library (None = UNKNOWN) and of the library the
                output is decoded as (None = UNKNOWN, the learned marginal library).
            target_library_emb: optional [D] / [B, D] embedding overriding target_library_id
                (e.g. the mean embedding of all cell libraries).

        Returns decoded output (log1p scale).
        """
        with torch.no_grad():
            # Encode with source modality signal
            source_emb = self._get_modality_emb(x_source.size(0), source_modality_id,
                                                x_source.device)
            s_source = s_target = None
            if self.libsize_cond:
                source_log_lib = self.log_library_size(x_source)
                s_source = self._standardize_libsize(source_log_lib)
                if target_log_libsize is None:
                    s_target = s_source
                else:
                    s_target = self._standardize_libsize(
                        self._as_log_libsize_column(target_log_libsize, x_source.size(0), x_source))
            l_source = self._get_library_emb(x_source.size(0), x_source.device, library_id=source_library_id)
            encoded = self.encoder(self._encoder_input(x_source, source_emb, s_source, l_source))
            mu_latent = self.mu(encoded)
            # Deterministic: use mu directly (no sampling)
            target_emb = self._get_modality_emb(x_source.size(0), target_modality_id,
                                                x_source.device)
            l_target = self._get_library_emb(x_source.size(0), x_source.device, library_id=target_library_id,
                                             library_emb=target_library_emb)
            decoded = self.decoder(self._decoder_input(mu_latent, target_emb, s_target, l_target))
        return decoded

    def decode_cross_train(self, mu_source, target_modality_id, target_log_libsize=None,
                           target_library_id=None):
        """
        Cross-decode for training (with gradients).

        Takes pre-computed mu from source forward pass, decodes with target
        embedding. Gradients flow through decoder and modality embedding
        but not back through the encoder (mu_source is detached).

        Args:
            mu_source: [B, L] latent mu from source modality
            target_modality_id: int, modality to decode as
            target_log_libsize: [B]/[B,1] log1p library size of the target samples
                (required when libsize_cond)

        Returns:
            decoded: [B, G] decoded output (log1p scale)
        """
        emb = self._get_modality_emb(mu_source.size(0), target_modality_id,
                                     mu_source.device)
        s = None
        if self.libsize_cond:
            if target_log_libsize is None:
                raise ValueError("decode_cross_train needs target_log_libsize when libsize_cond is on")
            s = self._standardize_libsize(
                self._as_log_libsize_column(target_log_libsize, mu_source.size(0), mu_source))
        lemb = self._get_library_emb(mu_source.size(0), mu_source.device, library_id=target_library_id,
                                     apply_dropout=self.training)
        decoded = self.decoder(self._decoder_input(mu_source.detach(), emb, s, lemb))
        return decoded

    def load_pretrained_partial(self, pretrained_path, device=None):
        """
        Load weights from a pretrained VAE_JSD model, expanding encoder
        layer 0 (if cond_encoder) and decoder layer 0 to accommodate the
        modality conditioning dimensions.

        The conditioning columns are zero-initialized so the model initially
        behaves identically to the pretrained model regardless of modality_id.
        """
        if device is None:
            device = next(self.parameters()).device
        pretrained = torch.load(pretrained_path, map_location=device)

        model_dict = self.state_dict()

        # Separate matched and mismatched keys
        matched = {}
        mismatched = {}
        missing_in_pretrained = []
        for k, v in model_dict.items():
            if k in pretrained:
                if pretrained[k].shape == v.shape:
                    matched[k] = pretrained[k]
                else:
                    mismatched[k] = pretrained[k]
            else:
                missing_in_pretrained.append(k)

        # Load all matched weights
        model_dict.update(matched)

        # Handle layer 0 weight expansion for both encoder and decoder
        for layer_key in ["encoder.0.weight", "decoder.0.weight"]:
            if layer_key in mismatched:
                old_w = mismatched[layer_key]
                new_w = model_dict[layer_key]
                old_dim = old_w.shape[1]
                new_w[:, :old_dim] = old_w
                new_w[:, old_dim:] = 0.0
                model_dict[layer_key] = new_w
                print(f"  {layer_key} expanded: {old_w.shape} -> {new_w.shape} "
                      f"(conditioning cols zero-initialized)")

        # Copy biases that match
        for bias_key in ["encoder.0.bias", "decoder.0.bias"]:
            if bias_key in pretrained and bias_key in model_dict:
                if pretrained[bias_key].shape == model_dict[bias_key].shape:
                    model_dict[bias_key] = pretrained[bias_key]

        self.load_state_dict(model_dict)

        print(f"  Loaded pretrained weights: {len(matched)} matched, "
              f"{len(mismatched)} expanded, {len(missing_in_pretrained)} new "
              f"({missing_in_pretrained})")
