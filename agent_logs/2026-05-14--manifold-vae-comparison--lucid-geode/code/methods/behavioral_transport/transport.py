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


def signed_step_vec(z_from: np.ndarray, z_to: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Per-dimension signed displacement, periodic shortest-arc where periods[d] > 0
    (0 = non-periodic dim). ``z_from``/``z_to`` are (d,); ``periods`` is (d,)."""
    d = np.asarray(z_to, dtype=np.float64) - np.asarray(z_from, dtype=np.float64)
    periods = np.asarray(periods, dtype=np.float64)
    for i, p in enumerate(periods):
        if p > 0:
            d[i] = (d[i] + p / 2.0) % p - p / 2.0
    return d


# ---------------------------------------------------------------------------
# Tangent vector field
# ---------------------------------------------------------------------------
class TransportField(nn.Module):
    """MLP tangent field. For a d-dimensional behavioral coordinate it outputs a
    ``(D x d)`` Jacobian per point; a behavioral step ``dz`` (d,) maps to an activation
    step ``Delta_h = J(h) @ dz`` (D,). d=1 reduces to a single tangent direction (the
    validated 1-D transport). Inputs standardized with stored (mean, std); deltas are in
    RAW ``h`` space so integrated paths live in the PCA coords the patcher expects.
    """

    def __init__(self, dim: int, hidden_dims: List[int], intrinsic_dim: int = 1):
        super().__init__()
        self.dim = dim
        self.intrinsic_dim = intrinsic_dim
        layers: List[nn.Module] = []
        d = dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers += [nn.Linear(d, dim * intrinsic_dim)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp(min=1e-6))

    def jacobian(self, h: torch.Tensor) -> torch.Tensor:
        """(B, dim, intrinsic_dim) tangent Jacobian at each point."""
        out = self.net((h - self.mean) / self.std)
        return out.view(h.shape[0], self.dim, self.intrinsic_dim)

    def step(self, h: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        """Activation step for a behavioral step ``dz`` (B, intrinsic_dim) -> (B, dim)."""
        return (self.jacobian(h) * dz.unsqueeze(1)).sum(dim=-1)


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


def _graph_neighbor_pairs(
    coords: np.ndarray, cls_idx: np.ndarray, adjacency: np.ndarray, periods: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(i, j, dz) example pairs for GRAPH-adjacent classes (d-dimensional ``dz`` =
    per-dim signed coord difference). ``coords`` is (W, d); ``adjacency`` is (W, W) 0/1."""
    W = coords.shape[0]
    by_class: dict[int, list] = {}
    for ex_i, c in enumerate(cls_idx.tolist()):
        by_class.setdefault(int(c), []).append(ex_i)
    ii, jj, dz = [], [], []
    for a in range(W):
        for b in range(W):
            if a >= b or adjacency[a, b] <= 0:
                continue
            if a not in by_class or b not in by_class:
                continue
            d_ab = signed_step_vec(coords[a], coords[b], periods)
            for ea in by_class[a]:
                for eb in by_class[b]:
                    ii.append(ea); jj.append(eb); dz.append(d_ab)
                    ii.append(eb); jj.append(ea); dz.append(-d_ab)
    return np.array(ii), np.array(jj), np.array(dz, dtype=np.float64)


def train_transport(
    features: torch.Tensor,          # (N, D) PCA-subspace activations
    cls_idx: torch.Tensor,           # (N,) class index per example
    ranks: np.ndarray,               # (W,) 1-D rank OR (W, d) coordinate per class
    periodic: bool,
    *,
    W: int,
    adjacency: np.ndarray | None = None,  # (W,W) 0/1 -> graph-neighbor pairs (d-dim mode)
    periods: np.ndarray | None = None,    # (d,) per-dim periods (0 = non-periodic); d-dim mode
    hidden_dims: List[int] = (256, 256),
    epochs: int = 300,
    lr: float = 1e-3,
    neighbor_radius: int = 1,
    w_density: float = 0.0,
    batch_size: int = 4096,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[TransportField, dict]:
    """Fit the tangent field real->real over near-in-behavior pairs. 1-D (rank + radius)
    when ``adjacency`` is None; d-dimensional (graph-adjacent, ``coords``+``adjacency``)
    otherwise. Field learns a (D x d) Jacobian; step = J @ dz."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)
    feats = features.to(dev).float()
    D = feats.shape[1]
    coords = np.asarray(ranks, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords[:, None]
    kdim = coords.shape[1]
    field = TransportField(D, list(hidden_dims), intrinsic_dim=kdim).to(dev)
    field.set_normalization(feats.mean(0), feats.std(0))

    cls_np = cls_idx.cpu().numpy()
    if adjacency is not None:
        if periods is None:
            periods = np.zeros(kdim)
        ii, jj, dz = _graph_neighbor_pairs(coords, cls_np, adjacency, periods)  # dz (n, d)
    else:
        ii, jj, dz1 = _neighbor_pairs(ranks, cls_np, periodic, W, neighbor_radius)
        dz = dz1[:, None]                                                       # (n, 1)
    if ii.size == 0:
        raise ValueError("transport: no neighbor pairs; check coords/adjacency.")
    ii_t = torch.from_numpy(ii).long()
    jj_t = torch.from_numpy(jj).long()
    dz_t = torch.from_numpy(dz).float().to(dev)                                 # (n, d)

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
            pred = field.step(hi, dz_t[idx])           # (B, D) via J @ dz
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
    return field, {"final_loss": history[-1] if history else None,
                   "n_pairs": int(n), "intrinsic_dim": int(kdim)}


@torch.no_grad()
def integrate_path(
    field: TransportField,
    c_from: torch.Tensor,    # (D,) start activation (real centroid)
    z_from,                  # scalar (1-D) or (d,) array
    z_to,
    *,
    n_steps: int,
    periodic: bool = False,
    period: float | None = None,
    periods: np.ndarray | None = None,   # (d,) per-dim periods (0=non-periodic); d-dim mode
) -> torch.Tensor:
    """Integrate the field from ``c_from`` toward ``z_to``. Returns (n_steps, D). Scalar
    z / ``period`` for the 1-D case; array z / ``periods`` for d-dim."""
    z_from = np.atleast_1d(np.asarray(z_from, dtype=np.float64))
    z_to = np.atleast_1d(np.asarray(z_to, dtype=np.float64))
    k = z_from.shape[0]
    if periods is None:
        p = float(period) if (periodic and period) else 0.0
        periods = np.array([p] * k)
    dz_total = signed_step_vec(z_from, z_to, periods)          # (d,)
    step = torch.tensor(dz_total / float(n_steps), dtype=torch.float32,
                        device=c_from.device).unsqueeze(0)      # (1, d)
    h = c_from.clone().float()
    path = [h.clone()]
    for _ in range(n_steps - 1):
        h = h + field.step(h.unsqueeze(0), step).squeeze(0)
        path.append(h.clone())
    return torch.stack(path, dim=0)
