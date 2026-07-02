"""Unit tests for behavioral transport (pure logic; no 8B, no I/O)."""

import math

import numpy as np
import torch

from methods.behavioral_transport.transport import (
    TransportField,
    discovered_order,
    integrate_path,
    signed_step,
    train_transport,
)


def _cycle_adj(n):
    a = np.zeros((n, n))
    for i in range(n):
        a[i, (i + 1) % n] = a[(i + 1) % n, i] = 1
    return a


def _line_adj(n):
    a = np.zeros((n, n))
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1
    return a


def _grid_adj(r, c):
    n = r * c
    a = np.zeros((n, n))
    for i in range(r):
        for j in range(c):
            k = i * c + j
            if j + 1 < c:
                a[k, k + 1] = a[k + 1, k] = 1
            if i + 1 < r:
                a[k, k + c] = a[k + c, k] = 1
    return a


def test_discovered_order_line():
    res = discovered_order(_line_adj(7))
    assert res is not None
    rank, periodic = res
    assert periodic is False
    # ranks are a permutation 0..6 (endpoints at the extremes)
    assert sorted(rank.tolist()) == list(range(7))


def test_discovered_order_cycle():
    res = discovered_order(_cycle_adj(7))
    assert res is not None
    rank, periodic = res
    assert periodic is True
    assert sorted(rank.tolist()) == list(range(7))


def test_discovered_order_branched_is_none():
    # a 2-D grid is dense (mean degree ~3, degree-4 interior) -> not a 1-D chain
    assert discovered_order(_grid_adj(4, 4)) is None


def test_discovered_order_noisy_line_recovers_via_lcc():
    # 8-node line + 1 spurious chord + a disconnected 2-node fragment (like the
    # over-pruned alphabet graph). Robust order should use the largest component
    # and return a line (periodic=False) covering the 8 main nodes.
    n = 10
    a = np.zeros((n, n))
    for i in range(7):  # nodes 0..7 form a line
        a[i, i + 1] = a[i + 1, i] = 1
    a[1, 4] = a[4, 1] = 1          # spurious chord
    a[8, 9] = a[9, 8] = 1          # separate 2-node fragment
    res = discovered_order(a)
    assert res is not None
    rank, periodic = res
    assert periodic is False
    main = [i for i in range(8) if rank[i] >= 0]
    assert len(main) == 8              # all line nodes ranked
    assert rank[8] == -1 and rank[9] == -1   # fragment excluded
    # endpoints of the line are at the extreme ranks
    assert {int(rank[0]), int(rank[7])} == {0, 7} or {int(rank[0]), int(rank[7])} == {7, 0}


def test_signed_step_periodic_shortest_arc():
    # on a period-7 ring, 0 -> 6 is -1 (shortest), not +6
    assert abs(signed_step(0.0, 6.0, True, 7.0) - (-1.0)) < 1e-9
    assert abs(signed_step(0.0, 3.0, False, 7.0) - 3.0) < 1e-9


def test_integrate_path_shape_and_endpoints():
    torch.manual_seed(0)
    field = TransportField(dim=5, hidden_dims=[16])
    c0 = torch.zeros(5)
    path = integrate_path(field, c0, 0.0, 4.0, n_steps=10, periodic=False, period=7.0)
    assert path.shape == (10, 5)
    assert torch.allclose(path[0], c0)  # starts exactly at the real endpoint


def test_train_transport_learns_a_line():
    # Synthetic: classes on a straight line in R^3, activations = class point + noise.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    W, per = 6, 30
    centers = np.stack([np.linspace(0, 5, W), np.zeros(W), np.zeros(W)], axis=1)  # x-axis line
    feats, cls = [], []
    for c in range(W):
        feats.append(centers[c] + 0.02 * rng.standard_normal((per, 3)))
        cls += [c] * per
    features = torch.tensor(np.concatenate(feats), dtype=torch.float32)
    cls_idx = torch.tensor(cls, dtype=torch.long)
    rank = np.arange(W)  # already ordered
    field, info = train_transport(
        features, cls_idx, rank, periodic=False, W=W,
        hidden_dims=[64, 64], epochs=150, neighbor_radius=1, seed=0,
    )
    assert info["n_pairs"] > 0
    # Integrate from class 0 centroid toward class 5: should move along +x by ~5.
    c0 = features[cls_idx == 0].mean(0)
    path = integrate_path(field, c0, 0.0, 5.0, n_steps=25, periodic=False, period=6.0)
    moved = (path[-1] - path[0]).numpy()
    assert moved[0] > 3.5, f"expected ~+5 along x, got {moved}"      # tracks the line
    assert abs(moved[1]) < 0.6 and abs(moved[2]) < 0.6, f"drifted off-axis: {moved}"


def test_train_transport_2d_plane_and_factored_control():
    # 2-D grid: activation = [row, col, 0] + noise; graph = 4-connected. The field
    # should learn a Jacobian s.t. moving one coordinate moves that axis only.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    R = C = 4
    per = 20
    node = lambda r, c: r * C + c
    coords, feats, cls = [], [], []
    for r in range(R):
        for c in range(C):
            coords.append([float(r), float(c)])
            feats.append(np.array([r, c, 0.0]) + 0.02 * rng.standard_normal((per, 3)))
            cls += [node(r, c)] * per
    coords = np.array(coords)
    features = torch.tensor(np.concatenate(feats), dtype=torch.float32)
    cls_idx = torch.tensor(cls, dtype=torch.long)
    W = R * C
    A = np.zeros((W, W))
    for r in range(R):
        for c in range(C):
            if c + 1 < C:
                A[node(r, c), node(r, c + 1)] = A[node(r, c + 1), node(r, c)] = 1
            if r + 1 < R:
                A[node(r, c), node(r + 1, c)] = A[node(r + 1, c), node(r, c)] = 1
    field, info = train_transport(
        features, cls_idx, coords, periodic=False, W=W, adjacency=A,
        periods=np.array([0.0, 0.0]), hidden_dims=[64, 64], epochs=250, seed=0,
    )
    assert info["intrinsic_dim"] == 2 and info["n_pairs"] > 0
    c00 = features[cls_idx == node(0, 0)].mean(0)
    # move +row -> x rises ~3, col/z ~0 (factored)
    p_row = integrate_path(field, c00, [0.0, 0.0], [3.0, 0.0], n_steps=25, periods=np.array([0.0, 0.0]))
    m = (p_row[-1] - p_row[0]).numpy()
    assert m[0] > 2.0 and abs(m[1]) < 0.7 and abs(m[2]) < 0.7, m
    # move +col -> y rises ~3, row ~0 (factored control)
    p_col = integrate_path(field, c00, [0.0, 0.0], [0.0, 3.0], n_steps=25, periods=np.array([0.0, 0.0]))
    m2 = (p_col[-1] - p_col[0]).numpy()
    assert m2[1] > 2.0 and abs(m2[0]) < 0.7, m2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL PASSED")
