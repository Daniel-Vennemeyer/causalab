"""Manifold protocol adapter for trained behavior-aligned VAEs.

``VAEManifold`` wraps an ``ActivationVAE`` or ``AtlasVAE`` plus standardization
buffers so the trained VAE is a drop-in for the spline/flow manifolds: it
exposes ``encode``/``decode``/``project``/``encode_to_nearest_point``/
``make_steering_grid`` with matching shapes, plus a plain-dict
``state_dict_to_save``/``from_state_dict`` round-trip (the analysis layer
serializes the dict; this method performs no disk I/O).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn

from .models import ActivationVAE, AtlasVAE


_EPS = 1e-6


class VAEManifold(nn.Module):
    """Standardization-aware manifold adapter around a VAE.

    Args:
        vae: a trained ``ActivationVAE`` or ``AtlasVAE``.
        mean: standardization mean (ambient_dim,).
        std: standardization std (ambient_dim,).
        eps: numerical-stability epsilon.
    """

    def __init__(
        self,
        vae: nn.Module,
        mean: Tensor,
        std: Tensor,
        eps: float = _EPS,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.register_buffer("_mean", mean)
        self._mean: Tensor
        self.register_buffer("_std", std)
        self._std: Tensor
        self._eps = eps

    # ----- protocol properties ------------------------------------------------
    @property
    def intrinsic_dim(self) -> int:
        return self.vae.intrinsic_dim

    @property
    def ambient_dim(self) -> int:
        return self.vae.ambient_dim

    @property
    def k(self) -> int:
        return self.vae.intrinsic_dim

    @property
    def n(self) -> int:
        return self.vae.ambient_dim

    @property
    def topology(self) -> str:
        return self.vae.topology

    @property
    def periodic_dims(self) -> list[int]:
        return self.vae.adapter.periodic_dims

    @property
    def periods(self) -> list[float]:
        return self.vae.adapter.periods

    # ----- standardization ----------------------------------------------------
    def _standardize(self, x: Tensor) -> Tensor:
        return (x - self._mean) / (self._std + self._eps)

    def _unstandardize(self, x: Tensor) -> Tensor:
        return x * (self._std + self._eps) + self._mean

    # ----- encode / decode ----------------------------------------------------
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode ambient points to ``(u, residual)``.

        ``u`` is the intrinsic coordinate from the posterior MEAN; ``residual``
        is the per-row reconstruction-error norm (B, 1), matching spline's
        encode contract.
        """
        x = x.to(self._mean.device)
        x_norm = self._standardize(x)
        mu, _ = self.vae.encode_dist(x_norm)
        u = self.vae.mu_to_intrinsic(mu)
        x_hat = self.decode(u)
        residual = (x - x_hat).norm(dim=-1, keepdim=True)
        return u, residual

    def decode(self, u: Tensor, r: Tensor | None = None) -> Tensor:
        """Decode intrinsic coordinates to ambient space (unstandardized)."""
        u = u.to(self._mean.device)
        x_norm = self.vae.decode(u)
        return self._unstandardize(x_norm)

    def project(self, x: Tensor) -> Tensor:
        u, _ = self.encode(x)
        return self.decode(u)

    def encode_to_nearest_point(
        self,
        x: Tensor,
        n_iters: int = 0,
        tol: float = 1e-6,
        damping: float = 1e-6,
    ) -> Tuple[Tensor, Tensor]:
        """Project x onto the manifold.

        For a VAE the posterior mean IS the projection, so this returns the
        encoder-mean coordinates. The signature mirrors ``SplineManifold`` for
        drop-in compatibility; ``n_iters``/``tol``/``damping`` are accepted but
        unused. The residual here is the full off-manifold displacement VECTOR
        ``x - decode(u)`` (shape (B, ambient_dim)).
        """
        x = x.to(self._mean.device)
        x_norm = self._standardize(x)
        mu, _ = self.vae.encode_dist(x_norm)
        u = self.vae.mu_to_intrinsic(mu)
        residual = x - self.decode(u)
        return u, residual

    # ----- steering grid ------------------------------------------------------
    def make_steering_grid(
        self,
        n_points_per_dim: int = 11,
        range_min: float = -3.0,
        range_max: float = 3.0,
        ranges: Tuple[Tuple[float, float], ...] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Cartesian grid over intrinsic coordinates.

        Angular dims (s1, cylinder) use range [-pi, pi). Same signature as the
        spline/flow manifolds.
        """
        d = self.intrinsic_dim

        if ranges is not None:
            if len(ranges) != d:
                raise ValueError(
                    f"ranges has {len(ranges)} entries but intrinsic_dim is {d}"
                )
            dim_ranges = list(ranges)
        else:
            dim_ranges = [(range_min, range_max)] * d
            for pd in self.periodic_dims:
                dim_ranges[pd] = (-math.pi, math.pi)

        # endpoint=False on angular dims to avoid duplicating -pi == pi.
        def _lin(lo: float, hi: float, periodic: bool) -> Tensor:
            if periodic:
                return torch.linspace(lo, hi, n_points_per_dim + 1)[:-1]
            return torch.linspace(lo, hi, n_points_per_dim)

        periodic_set = set(self.periodic_dims)

        if d == 1:
            coords = _lin(dim_ranges[0][0], dim_ranges[0][1], 0 in periodic_set)
            return coords.unsqueeze(-1)

        elif d == 2:
            coords0 = _lin(dim_ranges[0][0], dim_ranges[0][1], 0 in periodic_set)
            coords1 = _lin(dim_ranges[1][0], dim_ranges[1][1], 1 in periodic_set)
            u1, u2 = torch.meshgrid(coords0, coords1, indexing="ij")
            return torch.stack([u1.flatten(), u2.flatten()], dim=-1)

        else:
            grids = []
            for dim in range(d):
                coords = _lin(
                    dim_ranges[dim][0], dim_ranges[dim][1], dim in periodic_set
                )
                sparse = torch.zeros(coords.shape[0], d)
                sparse[:, dim] = coords
                grids.append(sparse)
            return torch.cat(grids, dim=0)

    # ----- flow-compatible fwd / inv (for ManifoldFeaturizer) -----------------
    def fwd(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Flow-compatible forward: ambient x -> (z, logdet).

        ``z`` is the intrinsic coordinate (B, intrinsic_dim) from the encoder
        mean; ``encode`` already handles internal standardization. Mirrors
        ``SplineManifold.fwd`` so this manifold is a drop-in for
        ``ManifoldFeaturizer``. The logdet is a zero placeholder (the VAE encode
        is not a volume-preserving bijection; downstream consumers ignore it).
        """
        u, _ = self.encode(x)
        return u, x.new_zeros(x.shape[0])

    def inv(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Flow-compatible inverse: intrinsic z -> (x, logdet).

        Accepts intrinsic-dim grid points (only the first ``intrinsic_dim``
        columns are used); ``decode`` un-standardizes to ambient PCA space.
        """
        u = z[:, : self.intrinsic_dim]
        return self.decode(u), z.new_zeros(z.shape[0])

    def get_config(self) -> Dict[str, Any]:
        """Minimal config for ``ManifoldFeaturizer.to_dict``."""
        return {
            "type": "vae",
            "intrinsic_dim": self.intrinsic_dim,
            "ambient_dim": self.ambient_dim,
        }

    def forward(self, x: Tensor) -> Tensor:
        return self.project(x)

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        self.vae = self.vae.to(*args, **kwargs)
        return self

    # ----- serialization (plain dict; analysis persists) ----------------------
    def _vae_config(self) -> Dict[str, Any]:
        vae = self.vae
        if isinstance(vae, AtlasVAE):
            return {
                "model_type": "atlas",
                "ambient_dim": vae.ambient_dim,
                "latent_dim": vae.latent_dim,
                "hidden_dims": list(self._hidden_dims),
                "topology": vae.topology,
                "n_charts": vae.n_charts,
            }
        assert isinstance(vae, ActivationVAE)
        return {
            "model_type": "flat",
            "ambient_dim": vae.ambient_dim,
            "latent_dim": vae.latent_dim,
            "hidden_dims": list(self._hidden_dims),
            "topology": vae.topology,
            "n_charts": None,
        }

    def set_hidden_dims(self, hidden_dims: list[int]) -> None:
        """Record the hidden-dims used to build the wrapped VAE (needed to
        reconstruct it from a state dict)."""
        self._hidden_dims = list(hidden_dims)

    def state_dict_to_save(self) -> Dict[str, Any]:
        if not hasattr(self, "_hidden_dims"):
            raise RuntimeError(
                "VAEManifold.set_hidden_dims must be called before "
                "state_dict_to_save so the architecture can be reconstructed"
            )
        cfg = self._vae_config()
        return {
            "config": cfg,
            "vae_state": {k: v.cpu() for k, v in self.vae.state_dict().items()},
            "mean": self._mean.cpu(),
            "std": self._std.cpu(),
            "eps": self._eps,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "VAEManifold":
        cfg = state["config"]
        if cfg["model_type"] == "atlas":
            vae: nn.Module = AtlasVAE(
                ambient_dim=cfg["ambient_dim"],
                latent_dim=cfg["latent_dim"],
                hidden_dims=cfg["hidden_dims"],
                topology=cfg["topology"],
                n_charts=cfg["n_charts"],
            )
        else:
            vae = ActivationVAE(
                ambient_dim=cfg["ambient_dim"],
                latent_dim=cfg["latent_dim"],
                hidden_dims=cfg["hidden_dims"],
                topology=cfg["topology"],
            )
        vae.load_state_dict(state["vae_state"])
        manifold = cls(
            vae=vae,
            mean=state["mean"],
            std=state["std"],
            eps=state.get("eps", _EPS),
        )
        manifold.set_hidden_dims(cfg["hidden_dims"])
        return manifold
