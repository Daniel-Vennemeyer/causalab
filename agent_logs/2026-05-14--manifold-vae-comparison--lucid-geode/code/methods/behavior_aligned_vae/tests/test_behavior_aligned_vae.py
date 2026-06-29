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


def test_loss_bundle_skips_zero_weight_terms():
    bundle = LossBundle(
        w_recon=1.0,
        w_kl=0.0,
        w_behavior=0.0,
        w_isometry=0.0,
        w_geodesic=0.0,
        w_patch=0.0,
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
    assert math.isfinite(total.item())
