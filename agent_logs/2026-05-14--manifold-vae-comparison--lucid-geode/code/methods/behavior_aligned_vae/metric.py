"""Differentiable Riemannian-metric estimation on VAE intrinsic coordinates.

Three metrics over the intrinsic manifold coordinates ``u`` (shape (B, k)):

- ``latent_linear``: the flat identity metric (geodesics are straight lines).
- ``decoder_pullback``: pull the ambient Euclidean metric back through the
  decoder, ``G = J^T J`` with ``J = d(decode)/d(u)``.
- ``behavior_pullback``: pull a behavior-space metric ``g_y`` back through the
  composition ``behavior(decode(u))``.

All functions return a batched metric tensor of shape (B, k, k). No disk I/O,
no hyperparameter defaults (numerical epsilons are fine).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor


DecodeFn = Callable[[Tensor], Tensor]
BehaviorFn = Callable[[Tensor], Tensor]


def _batched_jacobian(fn: Callable[[Tensor], Tensor], u: Tensor) -> Tensor:
    """Per-row Jacobian of ``fn`` at points ``u`` (B, k).

    Returns J of shape (B, out_dim, k) where ``out_dim`` is the output width of
    ``fn`` applied to a single row.
    """
    from torch.func import jacrev, vmap

    def single(u_row: Tensor) -> Tensor:
        return fn(u_row.unsqueeze(0)).squeeze(0)

    # vmap(jacrev(single)) gives (B, out_dim, k).
    return vmap(jacrev(single))(u)


def latent_linear_metric(u: Tensor) -> Tensor:
    """Flat identity metric, shape (B, k, k)."""
    b, k = u.shape
    eye = torch.eye(k, dtype=u.dtype, device=u.device)
    return eye.unsqueeze(0).expand(b, k, k).contiguous()


def decoder_pullback_metric(decode_fn: DecodeFn, u: Tensor) -> Tensor:
    """``G(u) = J^T J`` with ``J = d(decode_fn)/d(u)``. Shape (B, k, k)."""
    j = _batched_jacobian(decode_fn, u)  # (B, ambient, k)
    g = torch.einsum("bik,bil->bkl", j, j)  # (B, k, k)
    return g


def behavior_pullback_metric(
    decode_fn: DecodeFn,
    behavior_fn: BehaviorFn,
    u: Tensor,
    g_y: Optional[Tensor] = None,
) -> Tensor:
    """``G = J^T g_y J`` with ``J = d(behavior_fn(decode_fn(u)))/d(u)``.

    ``g_y`` is an optional behavior-space metric of shape (n_behavior,
    n_behavior); identity if None. Shape (B, k, k).
    """

    def composed(uu: Tensor) -> Tensor:
        return behavior_fn(decode_fn(uu))

    j = _batched_jacobian(composed, u)  # (B, n_behavior, k)
    if g_y is None:
        g = torch.einsum("bik,bil->bkl", j, j)
    else:
        g_y = g_y.to(dtype=u.dtype, device=u.device)
        g = torch.einsum("bik,ij,bjl->bkl", j, g_y, j)
    return g


def compute_metric(
    kind: str,
    u: Tensor,
    decode_fn: Optional[DecodeFn] = None,
    behavior_fn: Optional[BehaviorFn] = None,
    g_y: Optional[Tensor] = None,
) -> Tensor:
    """Dispatch to a metric by ``kind``.

    ``kind`` in {"latent_linear", "decoder_pullback", "behavior_pullback"}.
    """
    if kind == "latent_linear":
        return latent_linear_metric(u)
    if kind == "decoder_pullback":
        if decode_fn is None:
            raise ValueError("decoder_pullback metric requires decode_fn")
        return decoder_pullback_metric(decode_fn, u)
    if kind == "behavior_pullback":
        if decode_fn is None or behavior_fn is None:
            raise ValueError(
                "behavior_pullback metric requires decode_fn and behavior_fn"
            )
        return behavior_pullback_metric(decode_fn, behavior_fn, u, g_y=g_y)
    raise ValueError(
        f"unknown metric kind {kind!r}; expected one of "
        "'latent_linear', 'decoder_pullback', 'behavior_pullback'"
    )
