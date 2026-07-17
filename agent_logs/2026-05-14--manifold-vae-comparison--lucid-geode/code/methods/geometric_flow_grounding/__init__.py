"""Geometric Flow Grounding (Yu et al. 2026) for behavioral steering.

Neural Tangent Projection applied to the transport field: instead of integrating a free
ambient velocity (which drifts off-manifold and needs a density penalty / re-anchoring), we
integrate in the latent space of a learned state decoder and decode each step, so every step
lands on the decoder manifold by construction (GFG's tangent-bundle grounding).
"""

from .gfg import StateDecoder, ntp_integrate_path, train_state_decoder

__all__ = ["StateDecoder", "ntp_integrate_path", "train_state_decoder"]
