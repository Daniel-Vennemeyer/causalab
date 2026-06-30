"""Discretized geodesic solver over VAE intrinsic coordinates.

Approximates the shortest path between two points under a (possibly
position-dependent) Riemannian metric by minimizing the discrete path energy

    E = sum_i (du_i)^T G(midpoint_i) (du_i)

over the interior path points (endpoints fixed), with Adam. Periodic
dimensions are handled by initializing along the shortest arc and measuring
steps with wrap-aware differences.

All training knobs are required keyword arguments. No disk I/O.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
from torch import Tensor


MetricFn = Callable[[Tensor], Tensor]


def _wrap_delta(delta: Tensor, periodic_dims: List[int], periods: List[float]) -> Tensor:
    """Wrap differences on periodic dims into the shortest signed arc."""
    if not periodic_dims:
        return delta
    delta = delta.clone()
    for d, p in zip(periodic_dims, periods):
        delta[..., d] = (delta[..., d] + p / 2.0) % p - p / 2.0
    return delta


class GeodesicSolver:
    """Solver for discrete geodesics under a selectable metric."""

    def geodesic(
        self,
        u0: Tensor,
        u1: Tensor,
        *,
        metric_fn: MetricFn,
        n_points: int,
        n_iters: int,
        lr: float,
        periodic_dims: Optional[Sequence[int]] = None,
        periods: Optional[Sequence[float]] = None,
    ) -> Tensor:
        """Return a path of shape (n_points, k) from ``u0`` to ``u1``.

        Args:
            u0, u1: endpoints, shape (k,) or (1, k).
            metric_fn: maps (B, k) -> (B, k, k).
            n_points: number of path points including both endpoints (>= 2).
            n_iters: Adam iterations over the interior points.
            lr: Adam learning rate.
            periodic_dims: intrinsic dims that wrap.
            periods: matching periods for ``periodic_dims``.
        """
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")
        pdims: List[int] = list(periodic_dims) if periodic_dims else []
        pers: List[float] = list(periods) if periods else []

        u0 = u0.reshape(-1)
        u1 = u1.reshape(-1)
        k = u0.shape[0]
        device = u0.device
        dtype = u0.dtype

        # Shortest-arc displacement (wrap-aware on periodic dims).
        full_delta = _wrap_delta((u1 - u0).unsqueeze(0), pdims, pers).squeeze(0)

        ts = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
        # Interior points initialized on the (periodic) straight line.
        init = u0.unsqueeze(0) + ts.unsqueeze(-1) * full_delta.unsqueeze(0)
        interior = init[1:-1].clone().detach().requires_grad_(True)

        u0_fixed = u0.detach()
        u1_fixed = u1.detach()

        if interior.numel() == 0:
            # Only endpoints; nothing to optimize.
            return torch.stack([u0_fixed, u1_fixed], dim=0)

        opt = torch.optim.Adam([interior], lr=lr)

        def assemble() -> Tensor:
            return torch.cat(
                [u0_fixed.unsqueeze(0), interior, u1_fixed.unsqueeze(0)], dim=0
            )

        for _ in range(n_iters):
            opt.zero_grad()
            path = assemble()  # (n_points, k)
            steps = _wrap_delta(path[1:] - path[:-1], pdims, pers)  # (n_points-1, k)
            mids = path[:-1] + 0.5 * steps  # (n_points-1, k)
            g = metric_fn(mids)  # (n_points-1, k, k)
            energy = torch.einsum("ni,nij,nj->n", steps, g, steps).sum()
            # torch.autograd.backward (not energy.backward()) to bypass nnsight's
            # Tensor.backward monkeypatch — see note in behavior_aligned_vae.py.
            torch.autograd.backward(energy)
            opt.step()

        with torch.no_grad():
            path = assemble().detach()
        return path
