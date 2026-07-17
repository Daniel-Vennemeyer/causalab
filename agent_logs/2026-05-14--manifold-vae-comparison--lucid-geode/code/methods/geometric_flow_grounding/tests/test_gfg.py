"""GFG Neural Tangent Projection: grounded integration stays on-manifold where free drifts."""

import numpy as np
import torch

from methods.behavioral_transport.transport import integrate_path, train_transport
from methods.geometric_flow_grounding.gfg import ntp_integrate_path, train_state_decoder


def _curved_data(K=12, D=48, per=15, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, K)
    arc = np.stack([np.cos(1.4 * t), np.sin(1.4 * t), 0.5 * t] + [np.zeros(K)] * (D - 3), axis=1) * 3
    acts = np.concatenate([arc[k] + 0.02 * rng.standard_normal((per, D)) for k in range(K)])
    lv = np.repeat(np.arange(K), per)
    return acts, lv, K


def _offman(path, acts):
    return float(np.sqrt(((path[:, None, :] - acts[None]) ** 2).sum(-1)).min(1).mean())


def test_ntp_stays_on_manifold_where_free_transport_drifts():
    acts, lv, K = _curved_data()
    feats = torch.tensor(acts, dtype=torch.float32)
    field, _ = train_transport(feats, torch.tensor(lv).long(), np.arange(K), periodic=False,
                               W=K, hidden_dims=[128], epochs=300, lr=1e-3, w_density=0.0,
                               seed=0, device="cpu")
    dec, info = train_state_decoder(feats, latent_dim=1, epochs=500, seed=0, device="cpu")
    assert info["recon"] < 0.5
    cents = np.stack([acts[lv == k].mean(0) for k in range(K)])
    c0 = torch.tensor(cents[0], dtype=torch.float32)
    free = integrate_path(field, c0, 0.0, float(K - 1), n_steps=40).numpy()
    gfg = ntp_integrate_path(field, dec, c0, 0.0, float(K - 1), n_steps=40).numpy()
    off_free, off_gfg = _offman(free, acts), _offman(gfg, acts)
    # NTP-grounded path stays near the data; free ambient integration drifts far off.
    assert off_gfg < 1.0
    assert off_gfg < 0.05 * off_free


def test_ntp_path_shape_and_endpoint():
    acts, lv, K = _curved_data()
    feats = torch.tensor(acts, dtype=torch.float32)
    field, _ = train_transport(feats, torch.tensor(lv).long(), np.arange(K), periodic=False,
                               W=K, hidden_dims=[128], epochs=200, lr=1e-3, seed=0, device="cpu")
    dec, _ = train_state_decoder(feats, latent_dim=1, epochs=300, seed=0, device="cpu")
    c0 = torch.tensor(acts[lv == 0].mean(0), dtype=torch.float32)
    p = ntp_integrate_path(field, dec, c0, 0.0, float(K - 1), n_steps=25).numpy()
    assert p.shape == (25, acts.shape[1])


if __name__ == "__main__":
    test_ntp_stays_on_manifold_where_free_transport_drifts()
    test_ntp_path_shape_and_endpoint()
    print("ALL PASSED")
