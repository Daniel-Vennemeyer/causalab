"""Local behavioral transport: a tangent vector field on activation space.

Learn ``v_theta(h): R^D -> R^D`` such that stepping a real activation by
``(z_j - z_i) * v(h_i)`` lands near a real activation ``h_j`` at the target
behavior. Steering integrates the field from a real centroid: every step is
anchored to where data actually is, which keeps intermediate activations
on-distribution — the failure mode of global-decode steering (our ``w_manifold``
saga: an MLP decoder hallucinates off-manifold interpolants).

Contrast with the other methods:
  - spline (paper): fit a parametric manifold on a GROUND-TRUTH coordinate; steer
    along its geodesic. Needs the coordinate and a manifold fit.
  - VAE: global encoder/decoder; interpolate in latent, decode. Decoder drifts.
  - transport (this): NO decoder, NO parametric manifold. A local vector field on
    activation space, trained real->real; steering = integrate from a real point.

1-D behavior for now (weekdays ring / alphabet line). The multi-dim generalization
(abstract traits) replaces ``v_theta(h): R^D -> R^D`` with a Jacobian
``J_theta(h): R^D -> R^{D x d}`` and integrates a d-vector ``dz``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Discovered 1-D coordinate from the transition graph (pure graph traversal)
# ---------------------------------------------------------------------------
def discovered_order(adj: np.ndarray) -> Optional[Tuple[np.ndarray, bool]]:
    """Order the present classes along the discovered transition graph — ROBUST to
    the noisy near-chains real behavioral graphs produce (spurious edges, a few
    branch nodes, disconnected fragments).

    Returns ``(rank, periodic)`` where ``rank`` is a (W,) integer position along the
    chain (``-1`` for classes off the main component) and ``periodic`` is True for a
    cycle. Returns ``None`` only when the graph is genuinely NOT 1-D-like (dense /
    2-D lattice: mean degree high or many degree>3 nodes) — that is the atlas regime.

    Robustness: restrict to the LARGEST connected component; a clean ~2-regular loop
    with edges==nodes is a cycle (traverse -> integer positions); otherwise treat a
    sparse near-path (mean degree <= 2.6, max degree <= 3) as a LINE and order nodes
    by graph distance from one diameter endpoint (double-BFS), which is monotone
    along the chain and tolerant of short branches/chords. ``adj`` is the (denoised)
    0/1 symmetric adjacency.
    """
    from collections import Counter

    from scipy.sparse.csgraph import connected_components, shortest_path

    W = adj.shape[0]
    A = (np.asarray(adj) > 0).astype(np.float64)
    present = np.where(A.sum(axis=1) > 0)[0]
    if present.size < 2:
        return None

    # --- largest connected component (handles fragmentation from over-pruning) --
    _n_comp, labels = connected_components(A, directed=False)
    main = Counter(labels[present].tolist()).most_common(1)[0][0]
    nodes = [int(i) for i in present if labels[i] == main]
    if len(nodes) < 2:
        return None
    sub = A[np.ix_(nodes, nodes)]
    subdeg = sub.sum(axis=1)
    n_nodes = len(nodes)
    n_edges = int(sub.sum() // 2)
    rank = np.full(W, -1, dtype=np.int64)

    # --- clean single cycle: 2-regular, edges == nodes -------------------------
    if subdeg.max() <= 2 and (subdeg == 1).sum() == 0 and n_edges == n_nodes:
        order = [0]
        seen = {0}
        cur, prev = 0, -1
        while True:
            nxt = [j for j in range(n_nodes) if sub[cur, j] > 0 and j != prev and j not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            order.append(cur)
            seen.add(cur)
        if len(order) == n_nodes:
            for pos, k in enumerate(order):
                rank[nodes[k]] = pos
            return rank, True

    # --- sparse near-path -> LINE via diameter ordering ------------------------
    if float(subdeg.mean()) <= 2.6 and int(subdeg.max()) <= 3:
        d = shortest_path(sub, directed=False, unweighted=True)
        finite = np.where(np.isfinite(d), d, -1.0)
        a = int(np.unravel_index(np.argmax(finite), finite.shape)[0])  # diameter end
        da = shortest_path(sub, directed=False, unweighted=True, indices=a)
        far = float(da[np.isfinite(da)].max())
        coord = np.where(np.isfinite(da), da, far + 1.0)
        order = sorted(range(n_nodes), key=lambda k: (coord[k], nodes[k]))
        for pos, k in enumerate(order):
            rank[nodes[k]] = pos
        return rank, False

    return None  # dense / 2-D — 1-D transport does not apply (atlas is future work)


def signed_step(z_from: float, z_to: float, periodic: bool, period: float) -> float:
    """Signed 1-D displacement, taking the periodic shortest arc when cyclic."""
    d = z_to - z_from
    if periodic:
        d = (d + period / 2.0) % period - period / 2.0
    return d


# ---------------------------------------------------------------------------
# Tangent vector field
# ---------------------------------------------------------------------------
class TransportField(nn.Module):
    """MLP tangent field ``v_theta(h): R^D -> R^D``. Inputs are standardized with
    stored (mean, std); the predicted delta is returned in the RAW ``h`` space so
    integrated paths live in the same PCA-subspace coordinates the patcher expects.
    """

    def __init__(self, dim: int, hidden_dims: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        d = dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers += [nn.Linear(d, dim)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp(min=1e-6))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net((h - self.mean) / self.std)


def _neighbor_pairs(
    ranks: np.ndarray, cls_idx: np.ndarray, periodic: bool, W: int, radius: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(i, j, dz) example-index pairs whose class ranks are within ``radius`` steps
    (signed periodic-aware ``dz`` in rank units). Cross-class only (dz != 0)."""
    rank_of = {int(c): int(ranks[c]) for c in range(W) if ranks[c] >= 0}
    by_rank: dict[int, list] = {}
    for ex_i, c in enumerate(cls_idx.tolist()):
        r = rank_of.get(int(c))
        if r is not None:
            by_rank.setdefault(r, []).append(ex_i)
    n_ranks = len([r for r in ranks if r >= 0])
    ii, jj, dz = [], [], []
    for r_i, exs_i in by_rank.items():
        for step in range(1, radius + 1):
            r_j = (r_i + step) % n_ranks if periodic else r_i + step
            if r_j not in by_rank:
                continue
            d = signed_step(r_i, r_j, periodic, float(n_ranks))
            for a in exs_i:
                for b in by_rank[r_j]:
                    ii.append(a); jj.append(b); dz.append(d)
                    ii.append(b); jj.append(a); dz.append(-d)
    return np.array(ii), np.array(jj), np.array(dz, dtype=np.float64)


def train_transport(
    features: torch.Tensor,          # (N, D) PCA-subspace activations
    cls_idx: torch.Tensor,           # (N,) class index per example
    ranks: np.ndarray,               # (W,) discovered rank per class (-1 absent)
    periodic: bool,
    *,
    W: int,
    hidden_dims: List[int] = (256, 256),
    epochs: int = 300,
    lr: float = 1e-3,
    neighbor_radius: int = 1,
    w_density: float = 0.0,
    batch_size: int = 4096,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[TransportField, dict]:
    """Fit the tangent field real->real over near-in-behavior example pairs."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)
    feats = features.to(dev).float()
    D = feats.shape[1]
    field = TransportField(D, list(hidden_dims)).to(dev)
    field.set_normalization(feats.mean(0), feats.std(0))

    ii, jj, dz = _neighbor_pairs(ranks, cls_idx.cpu().numpy(), periodic, W, neighbor_radius)
    if ii.size == 0:
        raise ValueError("transport: no neighbor pairs; check discovered ranks.")
    ii_t = torch.from_numpy(ii).long()
    jj_t = torch.from_numpy(jj).long()
    dz_t = torch.from_numpy(dz).float().to(dev)

    opt = torch.optim.Adam(field.parameters(), lr=lr)
    n = ii.size
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        ep = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            hi = feats[ii_t[idx]]
            hj = feats[jj_t[idx]]
            d = dz_t[idx].unsqueeze(1)
            pred = d * field(hi)                       # (B, D) transport step
            loss = ((pred - (hj - hi)) ** 2).sum(-1).mean()
            if w_density > 0.0:
                stepped = hi + pred
                dmin = torch.cdist(stepped, feats).min(dim=1).values
                loss = loss + w_density * (dmin ** 2).mean()
            opt.zero_grad()
            torch.autograd.backward(loss)   # nnsight-safe (see memory)
            opt.step()
            ep += float(loss.detach())
            nb += 1
        history.append(ep / max(1, nb))
    return field, {"final_loss": history[-1] if history else None, "n_pairs": int(n)}


@torch.no_grad()
def integrate_path(
    field: TransportField,
    c_from: torch.Tensor,    # (D,) start activation (real centroid)
    z_from: float,
    z_to: float,
    *,
    n_steps: int,
    periodic: bool,
    period: float,
) -> torch.Tensor:
    """Integrate the field from ``c_from`` toward the behavior coordinate ``z_to``.
    Returns an ``(n_steps, D)`` activation path in PCA-subspace coordinates."""
    dz_total = signed_step(z_from, z_to, periodic, period)
    step = dz_total / float(n_steps)
    h = c_from.clone().float()
    path = [h.clone()]
    for _ in range(n_steps - 1):
        h = h + step * field(h.unsqueeze(0)).squeeze(0)
        path.append(h.clone())
    return torch.stack(path, dim=0)
