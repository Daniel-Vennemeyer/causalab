"""Local behavioral transport: a learned tangent vector field on activation space.

Steering by integrating a real-activation-grounded field instead of decoding
arbitrary latent points. See transport.py for the full rationale.
"""

from .transport import (
    TransportField,
    discovered_order,
    signed_step,
    signed_step_vec,
    train_transport,
    integrate_path,
)

__all__ = [
    "TransportField",
    "discovered_order",
    "signed_step",
    "signed_step_vec",
    "train_transport",
    "integrate_path",
]
