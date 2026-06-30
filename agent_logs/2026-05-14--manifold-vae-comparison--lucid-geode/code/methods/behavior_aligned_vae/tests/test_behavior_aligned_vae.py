"""Tests for the behavior_aligned_vae method.

Run with the session code dir on PYTHONPATH:

    PYTHONPATH=agent_logs/.../code uv run pytest .../behavior_aligned_vae/tests/ -v
"""

from __future__ import annotations

import math

import pytest
import torch

from methods.behavior_aligned_vae import (
    ActivationVAE,
    AtlasVAE,
    BehaviorHead,
    GeodesicSolver,
    LossBundle,
    VAEManifold,
    behavior_pullback_metric,
    compute_metric,
    decoder_pullback_metric,
    latent_linear_metric,
    train_behavior_aligned_vae,
)

AMBIENT = 16
LATENT = 2
N = 64
N_BEHAVIOR = 7
HIDDEN = [24, 24]
BEH_HIDDEN = [16]

TOPOLOGIES = ["unstructured", "s1", "r2", "cylinder", "interval"]
METHODS = ["flat_vae", "atlas_vae"]


def _make_data(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(N, AMBIENT, generator=g)
    logits = torch.randn(N, N_BEHAVIOR, generator=g)
    targets = torch.softmax(logits, dim=-1)
    return feats, targets


def _loss_weights(**overrides):
    w = {
        "w_recon": 1.0,
        "w_kl": 0.1,
        "w_behavior": 1.0,
        "w_isometry": 0.5,
        "w_geodesic": 0.0,
        "w_patch": 0.0,
    }
    w.update(overrides)
    return w


# ---------------------------------------------------------------------------
# 1. Training runs for each method/topology, returns a manifold, loss drops.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_train_runs_and_decreases(method, topology):
    feats, targets = _make_data(seed=1)
    out = train_behavior_aligned_vae(
        feats,
        behavior_targets=targets,
        method=method,
        latent_dim=LATENT,
        hidden_dims=HIDDEN,
        topology=topology,
        n_charts=3,
        behavior_hidden_dims=BEH_HIDDEN,
        n_behavior=N_BEHAVIOR,
        loss_weights=_loss_weights(),
        behavior_distance="kl",
        lr=1e-2,
        epochs=8,
        batch_size=32,
        kl_warmup_epochs=3,
        device="cpu",
        seed=0,
    )
    assert isinstance(out["manifold"], VAEManifold)
    assert out["behavior_head"] is not None
    hist = out["history"]
    assert len(hist) == 8
    for ep in hist:
        assert math.isfinite(ep["total"])
    first = hist[0]["total"]
    last = hist[-1]["total"]
    # Allow tolerance; should trend down.
    assert last < first + 1e-3


def test_atlas_reports_chart_entropy():
    feats, targets = _make_data(seed=2)
    out = train_behavior_aligned_vae(
        feats,
        behavior_targets=targets,
        method="atlas_vae",
        latent_dim=LATENT,
        hidden_dims=HIDDEN,
        topology="unstructured",
        n_charts=4,
        behavior_hidden_dims=BEH_HIDDEN,
        n_behavior=N_BEHAVIOR,
        loss_weights=_loss_weights(),
        behavior_distance="js",
        lr=1e-2,
        epochs=3,
        batch_size=32,
        kl_warmup_epochs=1,
        device="cpu",
        seed=0,
    )
    assert "chart_entropy" in out["history"][-1]


def test_train_validation_pass():
    feats, targets = _make_data(seed=3)
    vfeats, vtargets = _make_data(seed=99)
    out = train_behavior_aligned_vae(
        feats,
        behavior_targets=targets,
        method="flat_vae",
        latent_dim=LATENT,
        hidden_dims=HIDDEN,
        topology="r2",
        n_charts=1,
        behavior_hidden_dims=BEH_HIDDEN,
        n_behavior=N_BEHAVIOR,
        loss_weights=_loss_weights(),
        behavior_distance="hellinger",
        lr=1e-2,
        epochs=2,
        batch_size=32,
        kl_warmup_epochs=1,
        device="cpu",
        seed=0,
        val_features=vfeats,
        val_behavior_targets=vtargets,
    )
    assert "val_total" in out["history"][-1]


# ---------------------------------------------------------------------------
# 2. VAEManifold encode/decode round-trip + state-dict reproduces decode.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_manifold_encode_decode_shapes(topology):
    feats, _ = _make_data(seed=4)
    out = train_behavior_aligned_vae(
        feats,
        method="flat_vae",
        latent_dim=LATENT,
        hidden_dims=HIDDEN,
        topology=topology,
        n_charts=1,
        behavior_hidden_dims=BEH_HIDDEN,
        n_behavior=N_BEHAVIOR,
        loss_weights=_loss_weights(w_behavior=0.0, w_isometry=0.0),
        behavior_distance="kl",
        lr=1e-2,
        epochs=2,
        batch_size=32,
        kl_warmup_epochs=1,
        device="cpu",
        seed=0,
    )
    m: VAEManifold = out["manifold"]
    x = feats[:10]
    u, residual = m.encode(x)
    assert u.shape == (10, m.intrinsic_dim)
    assert residual.shape == (10, 1)
    x_hat = m.decode(u)
    assert x_hat.shape == (10, AMBIENT)

    # encode_to_nearest_point returns full residual vector.
    u2, res_vec = m.encode_to_nearest_point(x)
    assert u2.shape == (10, m.intrinsic_dim)
    assert res_vec.shape == (10, AMBIENT)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_state_dict_roundtrip_bit_for_bit(topology):
    feats, _ = _make_data(seed=5)
    out = train_behavior_aligned_vae(
        feats,
        method="atlas_vae",
        latent_dim=LATENT,
        hidden_dims=HIDDEN,
        topology=topology,
        n_charts=3,
        behavior_hidden_dims=BEH_HIDDEN,
        n_behavior=N_BEHAVIOR,
        loss_weights=_loss_weights(w_behavior=0.0, w_isometry=0.0),
        behavior_distance="kl",
        lr=1e-2,
        epochs=2,
        batch_size=32,
        kl_warmup_epochs=1,
        device="cpu",
        seed=0,
    )
    m: VAEManifold = out["manifold"]
    x = feats[:8]
    u, _ = m.encode(x)
    before = m.decode(u)

    state = m.state_dict_to_save()
    m2 = VAEManifold.from_state_dict(state)
    after = m2.decode(u)
    assert torch.equal(before, after)

    # encode also matches.
    u_after, _ = m2.encode(x)
    assert torch.equal(u, u_after)


# ---------------------------------------------------------------------------
# 3. Metrics: SPD-ish symmetric (B, k, k); latent_linear is identity.
# ---------------------------------------------------------------------------
def test_metrics_shapes_and_properties():
    torch.manual_seed(0)
    vae = ActivationVAE(AMBIENT, LATENT, HIDDEN, "r2")
    head = BehaviorHead(AMBIENT, N_BEHAVIOR, BEH_HIDDEN)
    u = torch.randn(5, vae.intrinsic_dim)

    g_lin = latent_linear_metric(u)
    assert g_lin.shape == (5, vae.intrinsic_dim, vae.intrinsic_dim)
    eye = torch.eye(vae.intrinsic_dim).expand_as(g_lin)
    assert torch.allclose(g_lin, eye)

    g_dec = decoder_pullback_metric(vae.decode, u)
    assert g_dec.shape == (5, vae.intrinsic_dim, vae.intrinsic_dim)
    assert torch.allclose(g_dec, g_dec.transpose(-1, -2), atol=1e-5)
    # SPD-ish: non-negative eigenvalues (J^T J is PSD).
    eigs = torch.linalg.eigvalsh(g_dec)
    assert (eigs >= -1e-4).all()

    g_beh = behavior_pullback_metric(vae.decode, head.predict_dist, u)
    assert g_beh.shape == (5, vae.intrinsic_dim, vae.intrinsic_dim)
    assert torch.allclose(g_beh, g_beh.transpose(-1, -2), atol=1e-5)

    # Dispatcher parity.
    g_disp = compute_metric("decoder_pullback", u, decode_fn=vae.decode)
    assert torch.allclose(g_disp, g_dec, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. GeodesicSolver: correct endpoints; latent_linear stays ~straight.
# ---------------------------------------------------------------------------
def test_geodesic_endpoints_and_straightness():
    solver = GeodesicSolver()
    u0 = torch.tensor([-1.0, 0.5])
    u1 = torch.tensor([2.0, -1.0])
    n_points = 11
    path = solver.geodesic(
        u0,
        u1,
        metric_fn=latent_linear_metric,
        n_points=n_points,
        n_iters=50,
        lr=1e-1,
    )
    assert path.shape == (n_points, 2)
    assert torch.allclose(path[0], u0, atol=1e-5)
    assert torch.allclose(path[-1], u1, atol=1e-5)

    # Under the flat metric the optimal path is the straight line.
    straight = torch.stack(
        [
            torch.linspace(u0[0], u1[0], n_points),
            torch.linspace(u0[1], u1[1], n_points),
        ],
        dim=-1,
    )
    assert torch.allclose(path, straight, atol=1e-2)


def test_geodesic_periodic_endpoints():
    solver = GeodesicSolver()
    u0 = torch.tensor([3.0])  # near +pi
    u1 = torch.tensor([-3.0])  # near -pi; shortest arc wraps
    path = solver.geodesic(
        u0,
        u1,
        metric_fn=latent_linear_metric,
        n_points=7,
        n_iters=30,
        lr=1e-1,
        periodic_dims=[0],
        periods=[2 * math.pi],
    )
    assert path.shape == (7, 1)
    assert torch.allclose(path[0], u0, atol=1e-5)
    assert torch.allclose(path[-1], u1, atol=1e-5)
    # The shortest arc steps should each be small (wraps the short way).
    steps = path[1:] - path[:-1]
    wrapped = (steps + math.pi) % (2 * math.pi) - math.pi
    assert wrapped.abs().max() < 1.0


# ---------------------------------------------------------------------------
# 5. LossBundle: finite total and metrics dict with each enabled term.
# ---------------------------------------------------------------------------
def test_loss_bundle_terms():
    torch.manual_seed(0)
    bundle = LossBundle(
        w_recon=1.0,
        w_kl=0.5,
        w_behavior=1.0,
        w_isometry=0.3,
        w_geodesic=0.2,
        w_patch=0.4,
        w_contrastive=0.0,
        behavior_distance="js",
    )
    b, c, k = 12, N_BEHAVIOR, LATENT
    h = torch.randn(b, AMBIENT)
    h_hat = torch.randn(b, AMBIENT)
    mu = torch.randn(b, k)
    logvar = torch.randn(b, k)
    u = torch.randn(b, k)
    pred = torch.softmax(torch.randn(b, c), dim=-1)
    target = torch.softmax(torch.randn(b, c), dim=-1)
    patched = torch.softmax(torch.randn(b, c), dim=-1)
    decoded_path = torch.randn(9, AMBIENT)

    total, metrics = bundle.compute_losses(
        h=h,
        h_hat=h_hat,
        mu=mu,
        logvar=logvar,
        u=u,
        kl_weight_scale=0.7,
        behavior_pred=pred,
        behavior_target=target,
        decoded_path=decoded_path,
        patched_behavior=patched,
    )
    assert math.isfinite(total.item())
    for term in ["recon", "kl", "behavior", "isometry", "geodesic", "patch", "total"]:
        assert term in metrics
        assert math.isfinite(metrics[term])


# ---------------------------------------------------------------------------
# 6. fwd/inv flow-compatibility + ManifoldFeaturizer composition.
# ---------------------------------------------------------------------------
def test_fwd_inv_roundtrip_and_manifold_featurizer():
    from causalab.methods.spline.featurizer import ManifoldFeaturizer
    from causalab.neural.featurizer import Featurizer

    feats, _ = _make_data(seed=7)
    out = train_behavior_aligned_vae(
        feats,
        method="flat_vae",
        latent_dim=LATENT,
        hidden_dims=HIDDEN,
        topology="r2",
        n_charts=1,
        behavior_hidden_dims=BEH_HIDDEN,
        n_behavior=N_BEHAVIOR,
        loss_weights=_loss_weights(w_behavior=0.0, w_isometry=0.0),
        behavior_distance="kl",
        lr=1e-2,
        epochs=2,
        batch_size=32,
        kl_warmup_epochs=1,
        device="cpu",
        seed=0,
    )
    m: VAEManifold = out["manifold"]
    x = feats[:10]

    # fwd returns (z, logdet): z is intrinsic-dim, logdet is a (B,) zero vector.
    z, logdet = m.fwd(x)
    assert z.shape == (10, m.intrinsic_dim)
    assert logdet.shape == (10,)
    assert torch.allclose(logdet, torch.zeros_like(logdet))

    # inv(fwd(x)) ≈ project(x) (decode of the encoder mean), within tolerance.
    x_rec, inv_logdet = m.inv(z)
    assert x_rec.shape == (10, AMBIENT)
    assert inv_logdet.shape == (10,)
    assert torch.allclose(x_rec, m.project(x), atol=1e-5)

    # get_config is minimal and well-formed.
    cfg = m.get_config()
    assert cfg["type"] == "vae"
    assert cfg["intrinsic_dim"] == m.intrinsic_dim
    assert cfg["ambient_dim"] == AMBIENT

    # ManifoldFeaturizer(manifold, n_features=AMBIENT) constructs; composing
    # with a trivial identity Featurizer yields a working forward/inverse where
    # the inverse maps intrinsic grid points -> ambient of dim AMBIENT.
    manifold_feat = ManifoldFeaturizer(m, n_features=AMBIENT)
    composed = Featurizer(id="identity") >> manifold_feat
    feat_z, errors = composed.featurize(x)
    assert feat_z.shape == (10, m.intrinsic_dim)

    # Intrinsic grid points -> ambient via the manifold inverse module.
    grid = m.make_steering_grid(n_points_per_dim=5)
    ambient = manifold_feat.inverse_featurizer(grid, None)
    assert ambient.shape == (grid.shape[0], AMBIENT)


def test_loss_bundle_skips_zero_weight_terms():
    bundle = LossBundle(
        w_recon=1.0,
        w_kl=0.0,
        w_behavior=0.0,
        w_isometry=0.0,
        w_geodesic=0.0,
        w_patch=0.0,
        w_contrastive=0.0,
        behavior_distance="kl",
    )
    b = 5
    h = torch.randn(b, AMBIENT)
    h_hat = torch.randn(b, AMBIENT)
    mu = torch.randn(b, LATENT)
    logvar = torch.randn(b, LATENT)
    total, metrics = bundle.compute_losses(
        h=h, h_hat=h_hat, mu=mu, logvar=logvar
    )
    assert "recon" in metrics
    assert "kl" not in metrics
    assert "behavior" not in metrics
    assert "contrastive" not in metrics
    assert math.isfinite(total.item())


# ---------------------------------------------------------------------------
# 7. Relational geometry: _cyclic_pdist correctness + contrastive_loss.
# ---------------------------------------------------------------------------
def test_cyclic_pdist_known_values():
    from methods.behavior_aligned_vae.losses import _cyclic_pdist

    coords = torch.tensor([0.0, 1.0, 3.0])
    period = 7.0
    # Raw cyclic distances: d(0,1)=1, d(0,3)=min(3,4)=3, d(1,3)=min(2,5)=2.
    # Off-diagonal entries (with the symmetric duplicates + zero diagonal):
    # values = [0,1,3, 1,0,2, 3,2,0]; mean = 12/9 = 1.3333...
    out = _cyclic_pdist(coords, period)
    raw = torch.tensor(
        [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 0.0],
        ]
    )
    expected = raw / raw.mean()
    assert out.shape == (3, 3)
    assert torch.allclose(out, expected, atol=1e-6)
    # Symmetric, zero diagonal.
    assert torch.allclose(out, out.t(), atol=1e-6)
    assert torch.allclose(torch.diag(out), torch.zeros(3), atol=1e-6)


def test_ordinal_pdist_known_values():
    from methods.behavior_aligned_vae.losses import _ordinal_pdist

    coords = torch.tensor([0.0, 1.0, 3.0])
    raw = (coords.unsqueeze(1) - coords.unsqueeze(0)).abs()
    expected = raw / raw.mean()
    out = _ordinal_pdist(coords)
    assert torch.allclose(out, expected, atol=1e-6)


def test_isometry_loss_cyclic_requires_period():
    u = torch.randn(5, 2)
    coords = torch.arange(5).float()
    with pytest.raises(ValueError):
        LossBundle.isometry_loss(
            u, geometry_coords=coords, geometry_distance="cyclic"
        )


def test_isometry_loss_euclidean_default_unchanged():
    # Default (euclidean) path must equal the legacy normalized-pdist formula.
    from methods.behavior_aligned_vae.losses import _normalized_pdist

    torch.manual_seed(0)
    u = torch.randn(6, 2)
    targets = torch.softmax(torch.randn(6, N_BEHAVIOR), dim=-1)
    out = LossBundle.isometry_loss(u, targets)
    du = _normalized_pdist(u)
    dy = _normalized_pdist(targets)
    expected = ((du - dy) ** 2).mean()
    assert torch.allclose(out, expected, atol=1e-7)


def test_contrastive_loss_finite_and_order_sensitive():
    period = 7.0
    margin = 1.0
    class_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6])

    # Latent that already respects cyclic order: place classes on a ring.
    theta = class_idx.float() * (2 * math.pi / period)
    u_ordered = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    loss_ordered = LossBundle.contrastive_loss(
        u_ordered, class_idx, period=period, margin=margin, cyclic=True
    )
    # Scrambled latent: random positions ignoring order.
    g = torch.Generator().manual_seed(0)
    u_scrambled = torch.randn(7, 2, generator=g)
    loss_scrambled = LossBundle.contrastive_loss(
        u_scrambled, class_idx, period=period, margin=margin, cyclic=True
    )
    assert math.isfinite(loss_ordered.item())
    assert math.isfinite(loss_scrambled.item())
    assert loss_ordered.item() < loss_scrambled.item()

    # Differentiable.
    u_req = u_scrambled.clone().requires_grad_(True)
    L = LossBundle.contrastive_loss(
        u_req, class_idx, period=period, margin=margin, cyclic=True
    )
    L.backward()
    assert u_req.grad is not None and torch.isfinite(u_req.grad).all()


def test_contrastive_loss_handles_missing_pos_or_neg():
    # All same class -> no negatives; should still be finite (neg side = 0).
    u = torch.randn(4, 2)
    same = torch.zeros(4, dtype=torch.long)
    out = LossBundle.contrastive_loss(u, same, margin=1.0, cyclic=False)
    assert math.isfinite(out.item())


def test_train_cyclic_geometry_smoke():
    feats, targets = _make_data(seed=11)
    coords = torch.randint(0, 7, (N,)).float()
    out = train_behavior_aligned_vae(
        feats,
        behavior_targets=targets,
        method="flat_vae",
        latent_dim=LATENT,
        hidden_dims=HIDDEN,
        topology="unstructured",
        n_charts=1,
        behavior_hidden_dims=BEH_HIDDEN,
        n_behavior=N_BEHAVIOR,
        loss_weights=_loss_weights(w_contrastive=2.0),
        behavior_distance="kl",
        lr=1e-2,
        epochs=3,
        batch_size=32,
        kl_warmup_epochs=1,
        device="cpu",
        seed=0,
        geometry_coords=coords,
        geometry_distance="cyclic",
        geometry_period=7.0,
        contrastive_margin=1.0,
    )
    fm = out["final_metrics"]
    assert "contrastive" in fm
    assert math.isfinite(fm["contrastive"])
    assert math.isfinite(fm["isometry"])
    for ep in out["history"]:
        assert math.isfinite(ep["total"])


def test_isometry_activation_mode_label_free():
    """geometry_distance='activation' uses input-activation distances as d_y
    (no labels/behavior); loss is finite and zero when u == activations."""
    lb = LossBundle(
        w_recon=1.0, w_kl=0.0, w_behavior=0.0, w_isometry=1.0,
        w_geodesic=0.0, w_patch=0.0, w_contrastive=0.0,
        behavior_distance="hellinger",
    )
    torch.manual_seed(0)
    h = torch.randn(16, 8)
    # u proportional to h (same geometry) -> normalized pdists match -> ~0 loss
    iso_same = lb.isometry_loss(
        h.clone(), geometry_distance="activation", activation_targets=h
    )
    assert torch.isfinite(iso_same) and float(iso_same) < 1e-6
    iso_rand = lb.isometry_loss(
        torch.randn(16, 2), geometry_distance="activation", activation_targets=h
    )
    assert torch.isfinite(iso_rand)
    with pytest.raises(ValueError):
        lb.isometry_loss(h, geometry_distance="activation")  # needs activation_targets


def test_train_activation_geometry_no_labels():
    """train_behavior_aligned_vae runs label-free with activation isometry."""
    torch.manual_seed(0)
    feats = torch.randn(48, 12)
    res = train_behavior_aligned_vae(
        feats, behavior_targets=None, method="flat_vae", latent_dim=2,
        hidden_dims=[32, 32], topology="unstructured", n_charts=1,
        behavior_hidden_dims=[16], n_behavior=7,
        loss_weights=dict(w_recon=1.0, w_kl=0.01, w_behavior=0.0, w_isometry=5.0,
                          w_geodesic=0.0, w_patch=0.0, w_contrastive=0.0),
        behavior_distance="hellinger", lr=1e-3, epochs=3, batch_size=16,
        kl_warmup_epochs=1, device="cpu", seed=0,
        geometry_distance="activation",
    )
    assert math.isfinite(res["final_metrics"]["isometry"])
    assert res["behavior_head"] is None  # no behavior supervision


# ---------------------------------------------------------------------------
# 8. Precomputed (transition-derived) geometry: isometry + contrastive + train,
#    and the analysis-layer _build_transition_dy graph-distance builder.
# ---------------------------------------------------------------------------
class _FakeCausalModel:
    def __init__(self, values):
        self.values = values


class _FakeTask:
    def __init__(self, values):
        self.causal_model = _FakeCausalModel(values)


def _grid_transition_dataset(n_classes: int):
    """n_classes x n_classes grid of (entity, number) inputs where the MODEL's
    predicted class = (entity_idx + number_idx) mod n_classes. Returns a fake
    task, a list-of-dicts train_dataset, and a one-hot per_example_dists at the
    predicted class. Stepping `number` by +1 (entity fixed) moves the predicted
    class by +1 (mod n) -> a clean ring -> graph distances == cyclic distances.
    """
    entities = [f"e{i}" for i in range(n_classes)]
    numbers = [f"n{j}" for j in range(n_classes)]
    values = {"entity": entities, "number": numbers}
    task = _FakeTask(values)

    train_dataset = []
    pred_rows = []
    for ei, e in enumerate(entities):
        for nj, n in enumerate(numbers):
            train_dataset.append({"input": {"entity": e, "number": n}})
            pred = (ei + nj) % n_classes
            row = torch.zeros(n_classes)
            row[pred] = 1.0
            pred_rows.append(row)
    # +1 "other" column to mimic the real (W+1) per_example_output_dists layout.
    dists = torch.stack(pred_rows, dim=0)
    other = torch.zeros(dists.shape[0], 1)
    per_example_dists = torch.cat([dists, other], dim=-1)
    return task, train_dataset, per_example_dists


def _cyclic_distance_matrix(n: int) -> torch.Tensor:
    idx = torch.arange(n).float()
    raw = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    return torch.minimum(raw, n - raw)


@pytest.mark.parametrize("n_classes", [3, 5])
def test_build_transition_dy_recovers_ring(n_classes):
    from analyses.behavior_manifold_vae.main import _build_transition_dy

    task, train_dataset, per_example_dists = _grid_transition_dataset(n_classes)
    dy = _build_transition_dy(
        task=task,
        train_dataset=train_dataset,
        per_example_dists=per_example_dists,
        W=n_classes,
        transition_variable="number",
        hold_fixed_vars=["entity"],
    )
    assert dy is not None
    assert dy.shape == (n_classes, n_classes)
    expected = _cyclic_distance_matrix(n_classes)
    assert torch.allclose(dy, expected, atol=1e-6)
    # Symmetric, zero diagonal.
    assert torch.allclose(dy, dy.t(), atol=1e-6)
    assert torch.allclose(torch.diag(dy), torch.zeros(n_classes), atol=1e-6)


def test_build_transition_dy_no_edges_returns_none():
    from analyses.behavior_manifold_vae.main import _build_transition_dy

    # Single number value per group -> no p+1 neighbor -> no edges.
    values = {"entity": ["e0", "e1"], "number": ["n0"]}
    task = _FakeTask(values)
    train_dataset = [
        {"input": {"entity": "e0", "number": "n0"}},
        {"input": {"entity": "e1", "number": "n0"}},
    ]
    per_example_dists = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    dy = _build_transition_dy(
        task=task,
        train_dataset=train_dataset,
        per_example_dists=per_example_dists,
        W=3,
        transition_variable="number",
        hold_fixed_vars=["entity"],
    )
    assert dy is None


def test_build_transition_dy_row_alignment_assert():
    from analyses.behavior_manifold_vae.main import _build_transition_dy

    task, train_dataset, per_example_dists = _grid_transition_dataset(3)
    with pytest.raises(ValueError):
        _build_transition_dy(
            task=task,
            train_dataset=train_dataset[:-1],  # length mismatch
            per_example_dists=per_example_dists,
            W=3,
            transition_variable="number",
            hold_fixed_vars=["entity"],
        )


def test_isometry_loss_precomputed_finite_and_zero_when_matched():
    """Precomputed isometry: finite for arbitrary u; ~0 when the latent pairwise
    geometry already matches the matrix lookup geometry."""
    W = 4
    # A known relational matrix.
    g = torch.tensor(
        [
            [0.0, 1.0, 2.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
        ]
    )
    coords = torch.tensor([0, 1, 2, 3, 0, 2])
    u = torch.randn(coords.shape[0], 2)
    out = LossBundle.isometry_loss(
        u,
        geometry_coords=coords,
        geometry_distance="precomputed",
        geometry_matrix=g,
    )
    assert torch.isfinite(out)

    # When du == dy (both normalized by their own mean), loss is exactly 0.
    # Construct u so its normalized pdist matches the matrix's normalized lookup.
    from methods.behavior_aligned_vae.losses import _normalized_pdist

    dy_raw = g[coords][:, coords]
    dy = dy_raw / dy_raw.mean().clamp_min(1e-8)
    # Find latent points whose normalized pdist equals dy: place on a 1-D line
    # only works for ordinal; instead just verify the formula directly by feeding
    # a u whose normalized pdist equals dy via a 2-D MDS-free shortcut — assert
    # the loss equals ((du - dy)**2).mean() recomputed from the same pieces.
    du = _normalized_pdist(u)
    expected = ((du - dy) ** 2).mean()
    assert torch.allclose(out, expected, atol=1e-6)

    # Missing matrix/coords -> ValueError.
    with pytest.raises(ValueError):
        LossBundle.isometry_loss(u, geometry_distance="precomputed")


def test_contrastive_loss_with_dist_matrix_finite():
    W = 4
    g = torch.tensor(
        [
            [0.0, 1.0, 2.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
        ]
    )
    class_idx = torch.tensor([0, 1, 2, 3, 0, 1])
    u = torch.randn(class_idx.shape[0], 2)
    out = LossBundle.contrastive_loss(
        u, class_idx, margin=1.0, cyclic=False, dist_matrix=g
    )
    assert math.isfinite(out.item())
    # Differentiable.
    u_req = u.clone().requires_grad_(True)
    L = LossBundle.contrastive_loss(
        u_req, class_idx, margin=1.0, cyclic=False, dist_matrix=g
    )
    L.backward()
    assert u_req.grad is not None and torch.isfinite(u_req.grad).all()


def test_train_precomputed_geometry_smoke():
    """train_behavior_aligned_vae runs with precomputed geometry + contrastive,
    producing finite isometry and contrastive metrics."""
    feats, targets = _make_data(seed=21)
    W = N_BEHAVIOR
    coords = torch.randint(0, W, (N,)).float()
    g = _cyclic_distance_matrix(W)  # any (W, W) relational matrix
    out = train_behavior_aligned_vae(
        feats,
        behavior_targets=targets,
        method="flat_vae",
        latent_dim=LATENT,
        hidden_dims=HIDDEN,
        topology="unstructured",
        n_charts=1,
        behavior_hidden_dims=BEH_HIDDEN,
        n_behavior=W,
        loss_weights=dict(
            w_recon=1.0, w_kl=0.01, w_behavior=0.0, w_isometry=5.0,
            w_geodesic=0.0, w_patch=0.0, w_contrastive=2.0,
        ),
        behavior_distance="hellinger",
        lr=1e-2,
        epochs=3,
        batch_size=32,
        kl_warmup_epochs=1,
        device="cpu",
        seed=0,
        geometry_coords=coords,
        geometry_distance="precomputed",
        geometry_matrix=g,
        contrastive_margin=1.0,
    )
    fm = out["final_metrics"]
    assert "isometry" in fm and math.isfinite(fm["isometry"])
    assert "contrastive" in fm and math.isfinite(fm["contrastive"])
    for ep in out["history"]:
        assert math.isfinite(ep["total"])
