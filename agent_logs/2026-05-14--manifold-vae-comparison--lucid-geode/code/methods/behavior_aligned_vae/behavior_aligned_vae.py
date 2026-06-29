"""Top-level training entry point for behavior-aligned activation-space VAEs.

``train_behavior_aligned_vae`` standardizes features, builds the requested model
(flat/metric -> ActivationVAE, atlas -> AtlasVAE) plus an optional behavior
head, runs an Adam training loop with KL warmup, and returns an in-memory dict
containing a ``VAEManifold`` adapter, the behavior head, per-epoch history, and
final metrics. No disk I/O happens here -- the analysis layer decides
persistence. All training knobs are required keyword arguments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from .losses import LossBundle
from .manifold import VAEManifold
from .models import ActivationVAE, AtlasVAE, BehaviorHead


_VALID_METHODS = ("flat_vae", "metric_vae", "atlas_vae")
_EPS = 1e-8


def _standardize_stats(x: Tensor) -> tuple[Tensor, Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    return mean, std


def _iter_batches(n: int, batch_size: int, generator: torch.Generator):
    perm = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        yield perm[start : start + batch_size]


def train_behavior_aligned_vae(
    features: Tensor,
    *,
    behavior_targets: Optional[Tensor] = None,
    method: str,
    latent_dim: int,
    hidden_dims: List[int],
    topology: str,
    n_charts: int,
    behavior_hidden_dims: List[int],
    n_behavior: int,
    loss_weights: Dict[str, float],
    behavior_distance: str,
    lr: float,
    epochs: int,
    batch_size: int,
    kl_warmup_epochs: int,
    device: str,
    seed: int,
    val_features: Optional[Tensor] = None,
    val_behavior_targets: Optional[Tensor] = None,
) -> Dict[str, Any]:
    """Train a behavior-aligned VAE and return in-memory artifacts.

    Args:
        features: train activations (N, ambient_dim).
        behavior_targets: optional train behavior probabilities (N, n_behavior).
        method: one of {"flat_vae", "metric_vae", "atlas_vae"}.
        latent_dim, hidden_dims, topology, n_charts: model architecture.
        behavior_hidden_dims, n_behavior: behavior head architecture.
        loss_weights: dict with keys w_recon, w_kl, w_behavior, w_isometry,
            w_geodesic, w_patch.
        behavior_distance: one of {"kl", "hellinger", "js"}.
        lr, epochs, batch_size, kl_warmup_epochs: optimization knobs.
        device, seed: runtime.
        val_features, val_behavior_targets: optional validation tensors.

    Returns:
        dict with keys: "manifold" (VAEManifold), "behavior_head"
        (BehaviorHead | None), "history" (list of per-epoch metric dicts),
        "final_metrics" (dict), "config" (dict).
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")

    torch.manual_seed(seed)
    dev = torch.device(device)
    gen = torch.Generator()
    gen.manual_seed(seed)

    features = features.to(dev).float()
    ambient_dim = features.shape[1]
    if behavior_targets is not None:
        behavior_targets = behavior_targets.to(dev).float()

    mean, std = _standardize_stats(features)
    features_norm = (features - mean) / (std + _EPS)

    if val_features is not None:
        val_features = val_features.to(dev).float()
        val_features_norm = (val_features - mean) / (std + _EPS)
        if val_behavior_targets is not None:
            val_behavior_targets = val_behavior_targets.to(dev).float()
    else:
        val_features_norm = None

    # Build model.
    if method == "atlas_vae":
        model: torch.nn.Module = AtlasVAE(
            ambient_dim=ambient_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            topology=topology,
            n_charts=n_charts,
        )
    else:  # flat_vae / metric_vae share the same architecture
        model = ActivationVAE(
            ambient_dim=ambient_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            topology=topology,
        )
    model = model.to(dev)

    # Behavior head only if needed.
    w_behavior = loss_weights["w_behavior"]
    w_patch = loss_weights["w_patch"]
    needs_behavior = (
        w_behavior != 0.0 or w_patch != 0.0 or behavior_targets is not None
    )
    behavior_head: Optional[BehaviorHead] = None
    if needs_behavior:
        behavior_head = BehaviorHead(
            in_dim=model.intrinsic_dim,
            n_behavior=n_behavior,
            hidden_dims=behavior_hidden_dims,
        ).to(dev)

    loss_bundle = LossBundle(
        w_recon=loss_weights["w_recon"],
        w_kl=loss_weights["w_kl"],
        w_behavior=w_behavior,
        w_isometry=loss_weights["w_isometry"],
        w_geodesic=loss_weights["w_geodesic"],
        w_patch=w_patch,
        behavior_distance=behavior_distance,
    )

    params = list(model.parameters())
    if behavior_head is not None:
        params += list(behavior_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    n = features_norm.shape[0]
    is_atlas = isinstance(model, AtlasVAE)

    def _forward(h_batch: Tensor):
        """Return (h_hat, mu, logvar, u, extra_loss, chart_entropy|None)."""
        if is_atlas:
            out = model(h_batch)
            return (
                out["h_hat"],
                out["mu"],
                out["logvar"],
                # intrinsic coords from primary chart's posterior mean
                model.mu_to_intrinsic(out["mu"]),
                None,
                out["chart_entropy"],
            )
        h_hat, mu, logvar, z_intrinsic = model(h_batch)
        return h_hat, mu, logvar, z_intrinsic, None, None

    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        if behavior_head is not None:
            behavior_head.train()
        # Linear KL warmup scale in [0, 1].
        if kl_warmup_epochs > 0:
            kl_scale = min(1.0, (epoch + 1) / kl_warmup_epochs)
        else:
            kl_scale = 1.0

        epoch_metrics: Dict[str, float] = {}
        n_batches = 0
        for idx in _iter_batches(n, batch_size, gen):
            h_batch = features_norm[idx]
            y_batch = (
                behavior_targets[idx] if behavior_targets is not None else None
            )

            opt.zero_grad()
            h_hat, mu, logvar, u, extra_loss, chart_entropy = _forward(h_batch)

            behavior_pred = None
            if behavior_head is not None and (
                loss_bundle.w_behavior != 0.0 or y_batch is not None
            ):
                behavior_pred = behavior_head.predict_dist(u)

            total, metrics = loss_bundle.compute_losses(
                h=h_batch,
                h_hat=h_hat,
                mu=mu,
                logvar=logvar,
                u=u,
                kl_weight_scale=kl_scale,
                behavior_pred=behavior_pred,
                behavior_target=y_batch,
                decoded_path=None,
                patched_behavior=None,
                extra_loss=extra_loss,
            )
            total.backward()
            opt.step()

            for key, val in metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + val
            if chart_entropy is not None:
                epoch_metrics["chart_entropy"] = epoch_metrics.get(
                    "chart_entropy", 0.0
                ) + chart_entropy.item()
            n_batches += 1

        for key in list(epoch_metrics):
            epoch_metrics[key] /= max(1, n_batches)
        epoch_metrics["epoch"] = float(epoch)
        epoch_metrics["kl_scale"] = kl_scale

        # Optional validation pass.
        if val_features_norm is not None:
            val_metrics = _evaluate(
                model,
                behavior_head,
                loss_bundle,
                val_features_norm,
                val_behavior_targets,
                is_atlas,
            )
            for key, val in val_metrics.items():
                epoch_metrics[f"val_{key}"] = val

        history.append(epoch_metrics)

    # Build manifold adapter.
    manifold = VAEManifold(vae=model, mean=mean.detach(), std=std.detach())
    manifold.set_hidden_dims(hidden_dims)

    final_metrics = dict(history[-1]) if history else {}

    config = {
        "method": method,
        "ambient_dim": ambient_dim,
        "latent_dim": latent_dim,
        "hidden_dims": list(hidden_dims),
        "topology": topology,
        "n_charts": n_charts,
        "behavior_hidden_dims": list(behavior_hidden_dims),
        "n_behavior": n_behavior,
        "loss_weights": dict(loss_weights),
        "behavior_distance": behavior_distance,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "kl_warmup_epochs": kl_warmup_epochs,
        "device": device,
        "seed": seed,
        "intrinsic_dim": model.intrinsic_dim,
    }

    return {
        "manifold": manifold,
        "behavior_head": behavior_head,
        "history": history,
        "final_metrics": final_metrics,
        "config": config,
    }


def _evaluate(
    model: torch.nn.Module,
    behavior_head: Optional[BehaviorHead],
    loss_bundle: LossBundle,
    h: Tensor,
    y: Optional[Tensor],
    is_atlas: bool,
) -> Dict[str, float]:
    model.eval()
    if behavior_head is not None:
        behavior_head.eval()
    with torch.no_grad():
        if is_atlas:
            out = model(h)
            h_hat, mu, logvar = out["h_hat"], out["mu"], out["logvar"]
            u = model.mu_to_intrinsic(mu)
        else:
            h_hat, mu, logvar, u = model(h)
        behavior_pred = None
        if behavior_head is not None and (
            loss_bundle.w_behavior != 0.0 or y is not None
        ):
            behavior_pred = behavior_head.predict_dist(u)
        _, metrics = loss_bundle.compute_losses(
            h=h,
            h_hat=h_hat,
            mu=mu,
            logvar=logvar,
            u=u,
            kl_weight_scale=1.0,
            behavior_pred=behavior_pred,
            behavior_target=y,
            decoded_path=None,
            patched_behavior=None,
        )
    return metrics
