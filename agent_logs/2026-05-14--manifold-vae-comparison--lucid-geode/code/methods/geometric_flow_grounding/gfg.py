"""Geometric Flow Grounding: Neural Tangent Projection for transport steering.

GFG (Yu et al. 2026) eliminates off-manifold integration drift by grounding dynamics on the
tangent bundle of a learned data manifold. Here the manifold is a small state decoder
G: Z(latent) -> X(activation subspace) (GFG's "state stream"), and the dynamics are the
already-validated transport tangent field v_theta (GFG's "dynamics stream"). The Neural Tangent
Projection is realized by LATENT-SPACE INTEGRATION AND DECODING (GFG Fig. 2): at each step we

  1) read the transport ambient velocity  v = field.step(h, dz),
  2) pull it back to a latent step via the decoder Jacobian (least squares: u_dot = J_G(u)^+ v),
  3) advance in latent  u <- u + u_dot  and DECODE  h = G(u).

Because h is always a decoder output, the trajectory stays on the manifold by construction --
the ambient velocity G(u+u_dot)-G(u) = J_G u_dot is tangent (Range J_G), so there is no normal
drift to accumulate. This is the principled replacement for transport's w_density + segment
re-anchoring (which we needed precisely because free ambient integration drifts).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from methods.behavioral_transport.transport import signed_step_vec


class StateDecoder(nn.Module):
    """Minimal GFG state stream: a symmetric autoencoder over the activation subspace.
    ``enc`` maps ambient -> latent; ``dec`` maps latent -> ambient (the generator G)."""

    def __init__(self, dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.enc(x)
        return self.dec(z), z


def train_state_decoder(
    features: torch.Tensor,          # (N, D) activation-subspace points
    latent_dim: int,
    *,
    hidden: int = 256,
    epochs: int = 800,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[StateDecoder, dict]:
    """Fit the state manifold G to reconstruct the data (GFG's L_topo, reconstruction only)."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    X = features.to(dev).float()
    model = StateDecoder(X.shape[1], latent_dim, hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last = float("nan")
    for _ in range(epochs):
        xh, _ = model(X)
        loss = ((xh - X) ** 2).sum(-1).mean()
        opt.zero_grad()
        torch.autograd.backward(loss)   # nnsight-safe
        opt.step()
        last = float(loss.detach())
    model.eval()
    return model, {"recon": last, "latent_dim": int(latent_dim)}


class VAEStateDecoder:
    """Adapt a trained behavior-aligned ``VAEManifold`` to the ``.enc``/``.dec`` interface GFG
    uses, so the higher-capacity VAE decoder can serve as the state manifold in place of the
    plain AE. ``enc`` returns the intrinsic coordinate (posterior mean); ``dec`` decodes it to
    unstandardized ambient space."""

    def __init__(self, manifold):
        self.manifold = manifold
        self.latent_dim = int(manifold.intrinsic_dim)

    def enc(self, x: torch.Tensor) -> torch.Tensor:
        u, _ = self.manifold.encode(x)
        return u

    def dec(self, z: torch.Tensor) -> torch.Tensor:
        return self.manifold.decode(z)


def _decoder_jvp_pullback(model, u: torch.Tensor, v_amb: torch.Tensor) -> torch.Tensor:
    """Least-squares latent step whose decoder JVP best matches the ambient velocity v_amb:
    u_dot = argmin_w ||J_G(u) w - v_amb|| = J_G(u)^+ v_amb.  Returns (latent_dim,)."""
    fn = lambda z: model.dec(z.unsqueeze(0)).squeeze(0)
    try:
        J = torch.autograd.functional.jacobian(fn, u, vectorize=True).detach()
    except Exception:  # some decoders (topology adapters) don't support vectorized jacobian
        J = torch.autograd.functional.jacobian(fn, u, vectorize=False).detach()
    sol = torch.linalg.lstsq(J, v_amb.unsqueeze(1)).solution   # (latent_dim, 1)
    return sol.squeeze(1)


def ntp_integrate_path(
    field,                           # trained TransportField (dynamics stream)
    model: StateDecoder,             # trained state decoder (manifold)
    c_from: torch.Tensor,            # (D,) start activation (real centroid)
    z_from,                          # scalar (1-D) or (d,) behavioral coordinate
    z_to,
    *,
    n_steps: int,
    periodic: bool = False,
    period: float | None = None,
    periods: np.ndarray | None = None,
) -> torch.Tensor:
    """GFG-grounded transport path: integrate the transport velocity in the decoder's LATENT
    space and decode each step, so the trajectory stays on the manifold by construction.
    Same signature as ``transport.integrate_path`` (drop-in) plus the decoder ``model``."""
    z_from = np.atleast_1d(np.asarray(z_from, dtype=np.float64))
    z_to = np.atleast_1d(np.asarray(z_to, dtype=np.float64))
    k = z_from.shape[0]
    if periods is None:
        p = float(period) if (periodic and period) else 0.0
        periods = np.array([p] * k)
    dz_total = signed_step_vec(z_from, z_to, periods)
    step = torch.tensor(dz_total / float(n_steps), dtype=torch.float32,
                        device=c_from.device).unsqueeze(0)      # (1, d) behavioral step
    u = model.enc(c_from.unsqueeze(0).float()).squeeze(0).detach()   # project start onto manifold
    h = model.dec(u.unsqueeze(0)).squeeze(0).detach()
    path = [h.clone()]
    for _ in range(n_steps - 1):
        v_amb = field.step(h.unsqueeze(0), step).squeeze(0).detach()     # transport velocity
        u = (u + _decoder_jvp_pullback(model, u, v_amb)).detach()        # latent-space step
        h = model.dec(u.unsqueeze(0)).squeeze(0).detach()                # decode -> on-manifold
        path.append(h.clone())
    return torch.stack(path, dim=0)
