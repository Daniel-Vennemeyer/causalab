"""Neural modules for behavior-aligned activation-space VAEs.

Contains the topology adapter (mapping unconstrained encoder coordinates to
intrinsic manifold coordinates and topology embeddings), the flat
``ActivationVAE``, the chart-routed ``AtlasVAE``, and a lightweight
``BehaviorHead`` predictor.

Layering: only ``torch``/``numpy`` and other ``causalab`` method primitives may
be imported here. No disk I/O. No baked-in training hyperparameters --
architectural constants (activation choice, numerical epsilons) are fine.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn


_VALID_TOPOLOGIES = ("unstructured", "s1", "interval", "r2", "cylinder")
_EPS = 1e-8


def _mlp(in_dim: int, hidden_dims: List[int], out_dim: int) -> nn.Sequential:
    """Build a simple MLP with GELU activations between linear layers."""
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.GELU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class TopologyAdapter:
    """Maps between unconstrained encoder coordinates, intrinsic coordinates,
    and topology embeddings consumed by a decoder.

    Conventions (per topology):

    - ``enc_dim``: size of the unconstrained Gaussian latent the encoder emits
      ``(mu, logvar)`` over and where KL is computed.
    - ``intrinsic_dim``: dimensionality of the manifold coordinates geodesics
      live in (what ``VAEManifold.encode`` returns).
    - ``embed_dim``: size of the (differentiable) coordinate the decoder
      consumes.

    Topologies:
    - ``unstructured``: enc == intrinsic == embed == latent_dim, identity maps.
    - ``s1``: enc_dim 2 (pre-angle Gaussian), intrinsic_dim 1 (angle theta via
      atan2), embed = [cos, sin].
    - ``interval``: enc_dim 1, intrinsic = embed = tanh(z) (bounded in (-1, 1)).
    - ``r2``: enc_dim 2, intrinsic == embed == identity.
    - ``cylinder``: S1 x R. enc_dim 3 (2 pre-angle + 1 linear), intrinsic_dim 2
      = (theta, r), embed = [cos theta, sin theta, r].
    """

    def __init__(self, latent_dim: int, topology: str):
        if topology not in _VALID_TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {_VALID_TOPOLOGIES}, got {topology!r}"
            )
        self.topology = topology
        self.latent_dim = latent_dim

        if topology == "unstructured":
            self.enc_dim = latent_dim
            self.intrinsic_dim = latent_dim
            self.embed_dim = latent_dim
        elif topology == "s1":
            self.enc_dim = 2
            self.intrinsic_dim = 1
            self.embed_dim = 2
        elif topology == "interval":
            self.enc_dim = 1
            self.intrinsic_dim = 1
            self.embed_dim = 1
        elif topology == "r2":
            self.enc_dim = 2
            self.intrinsic_dim = 2
            self.embed_dim = 2
        elif topology == "cylinder":
            self.enc_dim = 3
            self.intrinsic_dim = 2
            self.embed_dim = 3

    @property
    def periodic_dims(self) -> List[int]:
        """Indices into intrinsic coordinates that are angular (period 2*pi)."""
        if self.topology in ("s1", "cylinder"):
            return [0]
        return []

    @property
    def periods(self) -> List[float]:
        import math

        return [2.0 * math.pi for _ in self.periodic_dims]

    def raw_to_intrinsic(self, z_raw: Tensor) -> Tensor:
        """Map an unconstrained latent sample ``z_raw`` (B, enc_dim) to
        intrinsic manifold coordinates (B, intrinsic_dim)."""
        t = self.topology
        if t == "unstructured" or t == "r2":
            return z_raw
        if t == "s1":
            theta = torch.atan2(z_raw[:, 1], z_raw[:, 0]).unsqueeze(-1)
            return theta
        if t == "interval":
            return torch.tanh(z_raw)
        if t == "cylinder":
            theta = torch.atan2(z_raw[:, 1], z_raw[:, 0]).unsqueeze(-1)
            r = z_raw[:, 2:3]
            return torch.cat([theta, r], dim=-1)
        raise AssertionError(t)

    def intrinsic_to_embed(self, u: Tensor) -> Tensor:
        """Map intrinsic coordinates (B, intrinsic_dim) to the decoder's input
        embedding (B, embed_dim). Differentiable in ``u``."""
        t = self.topology
        if t == "unstructured" or t == "r2":
            return u
        if t == "s1":
            theta = u[:, 0]
            return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        if t == "interval":
            # u is already the bounded intrinsic value; identity embedding.
            return u
        if t == "cylinder":
            theta = u[:, 0]
            r = u[:, 1:2]
            return torch.cat(
                [torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1), r],
                dim=-1,
            )
        raise AssertionError(t)


class ActivationVAE(nn.Module):
    """MLP encoder/decoder VAE with a topology-aware latent space.

    The encoder maps an ambient activation ``h`` to a Gaussian ``q(z|h)`` in an
    UNCONSTRAINED coordinate space of size ``enc_dim`` (set by the topology).
    Intrinsic coordinates ``u`` are derived from a latent sample via the
    topology adapter, and the decoder consumes the topology embedding of ``u``.
    """

    def __init__(
        self,
        ambient_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        topology: str,
    ):
        super().__init__()
        self.ambient_dim = ambient_dim
        self.latent_dim = latent_dim
        self.topology = topology
        self.adapter = TopologyAdapter(latent_dim, topology)

        enc_dim = self.adapter.enc_dim
        embed_dim = self.adapter.embed_dim

        self.encoder_body = _mlp(ambient_dim, hidden_dims, hidden_dims[-1])
        self.to_mu = nn.Linear(hidden_dims[-1], enc_dim)
        self.to_logvar = nn.Linear(hidden_dims[-1], enc_dim)

        dec_hidden = list(reversed(hidden_dims))
        self.decoder = _mlp(embed_dim, dec_hidden, ambient_dim)

    @property
    def intrinsic_dim(self) -> int:
        return self.adapter.intrinsic_dim

    def encode_dist(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(mu, logvar)`` of ``q(z|h)`` in the unconstrained space."""
        feat = self.encoder_body(h)
        return self.to_mu(feat), self.to_logvar(feat)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample ``z_raw`` via the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def mu_to_intrinsic(self, mu: Tensor) -> Tensor:
        """Deterministic intrinsic coordinates from the posterior mean."""
        return self.adapter.raw_to_intrinsic(mu)

    def decode(self, z_intrinsic: Tensor) -> Tensor:
        """Decode intrinsic coordinates to a reconstructed activation."""
        embed = self.adapter.intrinsic_to_embed(z_intrinsic)
        return self.decoder(embed)

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(h_hat, mu, logvar, z_intrinsic)``."""
        mu, logvar = self.encode_dist(h)
        z_raw = self.reparameterize(mu, logvar)
        z_intrinsic = self.adapter.raw_to_intrinsic(z_raw)
        h_hat = self.decode(z_intrinsic)
        return h_hat, mu, logvar, z_intrinsic


class AtlasVAE(nn.Module):
    """Chart-routed VAE: a soft mixture of ``n_charts`` ActivationVAE charts.

    A router MLP maps ``h`` to softmax probabilities over charts. Each chart is
    an independent ``ActivationVAE``; the reconstruction is the router-weighted
    mixture of per-chart reconstructions.
    """

    def __init__(
        self,
        ambient_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        topology: str,
        n_charts: int,
    ):
        super().__init__()
        if n_charts < 1:
            raise ValueError(f"n_charts must be >= 1, got {n_charts}")
        self.ambient_dim = ambient_dim
        self.latent_dim = latent_dim
        self.topology = topology
        self.n_charts = n_charts

        self.router = _mlp(ambient_dim, hidden_dims, n_charts)
        self.charts = nn.ModuleList(
            [
                ActivationVAE(ambient_dim, latent_dim, hidden_dims, topology)
                for _ in range(n_charts)
            ]
        )
        self.adapter = self.charts[0].adapter

    @property
    def intrinsic_dim(self) -> int:
        return self.charts[0].intrinsic_dim

    def route(self, h: Tensor) -> Tensor:
        """Return router probabilities (B, n_charts)."""
        return torch.softmax(self.router(h), dim=-1)

    def encode_dist(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """Posterior of the most-likely (argmax-routed) chart, for protocol use."""
        probs = self.route(h)
        idx = probs.argmax(dim=-1)
        mus = []
        logvars = []
        for k, chart in enumerate(self.charts):
            mu_k, logvar_k = chart.encode_dist(h)
            mus.append(mu_k)
            logvars.append(logvar_k)
        mu_stack = torch.stack(mus, dim=1)  # (B, K, enc_dim)
        logvar_stack = torch.stack(logvars, dim=1)
        b = torch.arange(h.shape[0], device=h.device)
        return mu_stack[b, idx], logvar_stack[b, idx]

    def mu_to_intrinsic(self, mu: Tensor) -> Tensor:
        return self.adapter.raw_to_intrinsic(mu)

    def decode(self, z_intrinsic: Tensor) -> Tensor:
        """Decode via chart 0's decoder (deterministic protocol decode).

        The atlas mixture is only meaningful jointly with routing on a specific
        ``h``; for the manifold protocol's coordinate-only ``decode`` we use the
        primary chart so the map is well-defined on bare intrinsic coords.
        """
        return self.charts[0].decode(z_intrinsic)

    def forward(self, h: Tensor):
        """Return a dict with the soft-mixture reconstruction and per-chart parts.

        Keys: ``h_hat`` (B, ambient_dim), ``router_probs`` (B, K),
        ``mus`` (B, K, enc_dim), ``logvars`` (B, K, enc_dim),
        ``z_intrinsic`` (B, K, intrinsic_dim), ``chart_entropy`` (scalar),
        and primary-chart ``mu``/``logvar``/``z_primary`` for downstream use.
        """
        probs = self.route(h)  # (B, K)
        recons = []
        mus = []
        logvars = []
        zs = []
        for chart in self.charts:
            h_hat_k, mu_k, logvar_k, z_k = chart(h)
            recons.append(h_hat_k)
            mus.append(mu_k)
            logvars.append(logvar_k)
            zs.append(z_k)
        recon_stack = torch.stack(recons, dim=1)  # (B, K, ambient)
        h_hat = (probs.unsqueeze(-1) * recon_stack).sum(dim=1)

        mu_stack = torch.stack(mus, dim=1)
        logvar_stack = torch.stack(logvars, dim=1)
        z_stack = torch.stack(zs, dim=1)

        # Mean chart-usage entropy over the batch.
        mean_probs = probs.mean(dim=0)
        chart_entropy = -(mean_probs * (mean_probs + _EPS).log()).sum()

        idx = probs.argmax(dim=-1)
        b = torch.arange(h.shape[0], device=h.device)

        return {
            "h_hat": h_hat,
            "router_probs": probs,
            "mus": mu_stack,
            "logvars": logvar_stack,
            "z_intrinsic": z_stack,
            "chart_entropy": chart_entropy,
            "mu": mu_stack[b, idx],
            "logvar": logvar_stack[b, idx],
            "z_primary": z_stack[b, idx],
        }


class BehaviorHead(nn.Module):
    """MLP predictor from intrinsic coords or decoded activations to behavior.

    The caller chooses the input representation (intrinsic coordinates or
    decoded/ambient activations) and sets ``in_dim`` accordingly.
    """

    def __init__(self, in_dim: int, n_behavior: int, hidden_dims: List[int]):
        super().__init__()
        self.in_dim = in_dim
        self.n_behavior = n_behavior
        self.net = _mlp(in_dim, hidden_dims, n_behavior)

    def forward(self, x: Tensor) -> Tensor:
        """Return behavior logits (B, n_behavior)."""
        return self.net(x)

    def predict_dist(self, x: Tensor) -> Tensor:
        """Return behavior probabilities via softmax over logits."""
        return torch.softmax(self.forward(x), dim=-1)
