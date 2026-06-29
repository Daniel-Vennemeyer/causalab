"""Behavior-aligned activation-space VAE method (session-local).

Public API: a top-level training entry point, the model modules, the manifold
protocol adapter, the geodesic solver, the loss bundle, and the metric
estimators.
"""

from __future__ import annotations

from .behavior_aligned_vae import train_behavior_aligned_vae
from .geodesic import GeodesicSolver
from .losses import LossBundle
from .manifold import VAEManifold
from .metric import (
    behavior_pullback_metric,
    compute_metric,
    decoder_pullback_metric,
    latent_linear_metric,
)
from .models import ActivationVAE, AtlasVAE, BehaviorHead

__all__ = [
    "train_behavior_aligned_vae",
    "ActivationVAE",
    "AtlasVAE",
    "BehaviorHead",
    "VAEManifold",
    "GeodesicSolver",
    "LossBundle",
    "compute_metric",
    "latent_linear_metric",
    "decoder_pullback_metric",
    "behavior_pullback_metric",
]
