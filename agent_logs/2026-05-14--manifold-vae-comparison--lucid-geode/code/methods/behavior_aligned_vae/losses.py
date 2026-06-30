"""Composite loss bundle for behavior-aligned VAE training.

Computes a weighted sum of reconstruction, KL, behavior-alignment, isometry,
geodesic-naturalness, and patch/intervention-consistency terms. Each term is
gated by its weight; a zero weight skips the term entirely (and avoids any
expensive computation). All weights are required keyword arguments.

The method NEVER patches the frozen model -- ``patched_behavior`` is supplied
by the analysis layer. No disk I/O.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


_EPS = 1e-8
_VALID_BEHAVIOR_DISTANCES = ("kl", "hellinger", "js")


def _behavior_distance(p: Tensor, q: Tensor, kind: str) -> Tensor:
    """Mean per-row distance between two batches of probability vectors.

    ``p``, ``q`` are (B, C) probability tensors (rows sum to 1).
    """
    p = p.clamp_min(_EPS)
    q = q.clamp_min(_EPS)
    if kind == "kl":
        # KL(p || q), mean over batch.
        return (p * (p / q).log()).sum(dim=-1).mean()
    if kind == "hellinger":
        # Squared Hellinger distance, mean over batch.
        return 0.5 * ((p.sqrt() - q.sqrt()) ** 2).sum(dim=-1).mean()
    if kind == "js":
        m = 0.5 * (p + q)
        jsd = 0.5 * (p * (p / m).log()).sum(dim=-1) + 0.5 * (
            q * (q / m).log()
        ).sum(dim=-1)
        return jsd.mean()
    raise ValueError(
        f"behavior_distance must be one of {_VALID_BEHAVIOR_DISTANCES}, got {kind!r}"
    )


def _normalized_pdist(x: Tensor) -> Tensor:
    """Pairwise Euclidean distance matrix normalized by its mean (off-diagonal
    scale). Shape (B, B)."""
    d = torch.cdist(x, x, p=2)
    scale = d.mean().clamp_min(_EPS)
    return d / scale


def _cyclic_pdist(coords: Tensor, period: float) -> Tensor:
    """Pairwise cyclic distance matrix on 1-D ordinal positions, normalized by
    its mean (mirrors ``_normalized_pdist``). Shape (B, B).

    ``coords`` is (B,) or (B, 1). ``d_ij = min(|c_i - c_j|, period - |c_i - c_j|)``.
    """
    c = coords.reshape(-1).float()
    raw = (c.unsqueeze(1) - c.unsqueeze(0)).abs()
    d = torch.minimum(raw, period - raw)
    scale = d.mean().clamp_min(_EPS)
    return d / scale


def _ordinal_pdist(coords: Tensor) -> Tensor:
    """Pairwise absolute-difference distance matrix on 1-D ordinal positions,
    normalized by its mean (mirrors ``_normalized_pdist``). Shape (B, B)."""
    c = coords.reshape(-1).float()
    d = (c.unsqueeze(1) - c.unsqueeze(0)).abs()
    scale = d.mean().clamp_min(_EPS)
    return d / scale


class LossBundle:
    """Callable bundle of weighted VAE training losses.

    Args (all required): per-term weights and the behavior-distance kind.
    """

    def __init__(
        self,
        *,
        w_recon: float,
        w_kl: float,
        w_behavior: float,
        w_isometry: float,
        w_geodesic: float,
        w_patch: float,
        w_contrastive: float,
        behavior_distance: str,
    ):
        if behavior_distance not in _VALID_BEHAVIOR_DISTANCES:
            raise ValueError(
                f"behavior_distance must be one of {_VALID_BEHAVIOR_DISTANCES}, "
                f"got {behavior_distance!r}"
            )
        self.w_recon = w_recon
        self.w_kl = w_kl
        self.w_behavior = w_behavior
        self.w_isometry = w_isometry
        self.w_geodesic = w_geodesic
        self.w_patch = w_patch
        self.w_contrastive = w_contrastive
        self.behavior_distance = behavior_distance

    @staticmethod
    def recon_loss(h_hat: Tensor, h: Tensor) -> Tensor:
        """MSE summed over dims, mean over batch."""
        return ((h_hat - h) ** 2).sum(dim=-1).mean()

    @staticmethod
    def kl_loss(mu: Tensor, logvar: Tensor) -> Tensor:
        """Gaussian KL of q(z|h) vs N(0, I), mean over batch."""
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

    def behavior_loss(self, pred_probs: Tensor, target_probs: Tensor) -> Tensor:
        return _behavior_distance(pred_probs, target_probs, self.behavior_distance)

    @staticmethod
    def isometry_loss(
        u: Tensor,
        behavior_targets: Optional[Tensor] = None,
        *,
        geometry_coords: Optional[Tensor] = None,
        geometry_distance: str = "euclidean",
        geometry_period: Optional[float] = None,
        activation_targets: Optional[Tensor] = None,
        geometry_matrix: Optional[Tensor] = None,
    ) -> Tensor:
        """Match the normalized pairwise-distance geometry of intrinsic coords
        ``u`` to a target relational geometry ``d_y``.

        ``geometry_distance``:
          - ``"cyclic"``: ``d_y`` is the normalized cyclic distance on
            ``geometry_coords`` (requires ``geometry_period``).
          - ``"ordinal"``: ``d_y`` is the normalized |Δ| distance on
            ``geometry_coords``.
          - ``"activation"``: ``d_y`` is the normalized Euclidean pairwise
            distance between the input ``activation_targets`` (h). LABEL-FREE
            and behavior-free — preserves activation-space neighborhoods in the
            latent. Tests whether the behavioral geometry is already present in
            the activations (recoverable without any topology/label injection).
          - ``"precomputed"``: ``d_y`` is looked up from a (W, W) relational
            matrix ``geometry_matrix`` indexed by the per-example class indices
            ``geometry_coords`` (long/int (B,)). The matrix is built by the
            analysis layer (e.g. graph distance on the model's behavioral
            transitions) — NO topology/period/ordering is declared here; the
            geometry is whatever the matrix encodes. Normalized by its own mean,
            exactly like the other branches.
          - ``"euclidean"`` (default): ``d_y`` is the normalized Euclidean
            pairwise distance between ``behavior_targets`` (current behavior).
        """
        du = _normalized_pdist(u)
        if geometry_distance == "precomputed":
            if geometry_coords is None or geometry_matrix is None:
                raise ValueError(
                    "geometry_distance='precomputed' requires geometry_coords "
                    "(per-example class idx) and geometry_matrix ((W, W))"
                )
            coords = geometry_coords.reshape(-1).long()
            dy_raw = geometry_matrix[coords][:, coords]  # (B, B)
            dy = dy_raw / dy_raw.mean().clamp_min(_EPS)
            return ((du - dy) ** 2).mean()
        if geometry_distance == "cyclic" and geometry_coords is not None:
            if geometry_period is None:
                raise ValueError(
                    "geometry_distance='cyclic' requires geometry_period"
                )
            dy = _cyclic_pdist(geometry_coords, geometry_period)
        elif geometry_distance == "ordinal" and geometry_coords is not None:
            dy = _ordinal_pdist(geometry_coords)
        elif geometry_distance == "activation":
            if activation_targets is None:
                raise ValueError(
                    "geometry_distance='activation' requires activation_targets"
                )
            dy = _normalized_pdist(activation_targets)
        else:
            if behavior_targets is None:
                raise ValueError(
                    "euclidean isometry geometry requires behavior_targets"
                )
            dy = _normalized_pdist(behavior_targets)
        return ((du - dy) ** 2).mean()

    @staticmethod
    def contrastive_loss(
        u: Tensor,
        class_idx: Tensor,
        *,
        period: Optional[float] = None,
        margin: float,
        cyclic: bool = True,
        dist_matrix: Optional[Tensor] = None,
    ) -> Tensor:
        """Supervised-contrastive ordinal loss in RAW latent units.

        ``class_idx`` is (B,) int class positions. Pairwise class distance
        ``cd`` is taken from ``dist_matrix`` (a (W, W) RAW, integer-valued graph
        distance looked up by ``class_idx``) when supplied; otherwise it is
        cyclic (via ``period``) when ``cyclic`` else ``|Δ|``.
        Positives (``cd <= 1``: same/adjacent class) are pulled together
        (``du**2``); negatives (``cd >= 2``) are pushed past ``margin``
        (``relu(margin - du)**2``). ``du = torch.cdist(u, u)`` is NOT
        normalized — ``margin`` is in latent units. The diagonal is excluded;
        an absent positive/negative side contributes 0.
        """
        if dist_matrix is not None:
            idx = class_idx.reshape(-1).long()
            cd = dist_matrix[idx][:, idx]  # (B, B) RAW graph distance
        else:
            c = class_idx.reshape(-1).float()
            raw = (c.unsqueeze(1) - c.unsqueeze(0)).abs()
            if cyclic:
                if period is None:
                    raise ValueError("contrastive_loss cyclic=True requires period")
                cd = torch.minimum(raw, period - raw)
            else:
                cd = raw

        du = torch.cdist(u, u)
        b = u.shape[0]
        off_diag = ~torch.eye(b, dtype=torch.bool, device=u.device)
        pos_mask = (cd <= 1) & off_diag
        neg_mask = (cd >= 2) & off_diag

        pos_term = du.new_zeros(())
        if bool(pos_mask.any()):
            pos_term = (du[pos_mask] ** 2).sum()
        neg_term = du.new_zeros(())
        if bool(neg_mask.any()):
            neg_term = (torch.relu(margin - du[neg_mask]) ** 2).sum()

        n_off = off_diag.sum().clamp_min(1)
        return (pos_term + neg_term) / n_off

    @staticmethod
    def geodesic_loss(decoded_path: Tensor) -> Tensor:
        """Path-smoothness penalty: mean squared consecutive-step difference of
        decoded activations along a geodesic. ``decoded_path`` is (P, ambient)."""
        steps = decoded_path[1:] - decoded_path[:-1]
        return (steps**2).sum(dim=-1).mean()

    def compute_losses(
        self,
        *,
        h: Tensor,
        h_hat: Tensor,
        mu: Tensor,
        logvar: Tensor,
        u: Optional[Tensor] = None,
        kl_weight_scale: float = 1.0,
        behavior_pred: Optional[Tensor] = None,
        behavior_target: Optional[Tensor] = None,
        decoded_path: Optional[Tensor] = None,
        patched_behavior: Optional[Tensor] = None,
        extra_loss: Optional[Tensor] = None,
        geometry_coords: Optional[Tensor] = None,
        geometry_distance: str = "euclidean",
        geometry_period: Optional[float] = None,
        geometry_matrix: Optional[Tensor] = None,
        class_idx: Optional[Tensor] = None,
        contrastive_margin: float = 1.0,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Return ``(total_loss, metrics)``.

        Args:
            h, h_hat: target and reconstructed activations (B, ambient).
            mu, logvar: posterior parameters (B, enc_dim).
            u: intrinsic coordinates (B, k); required for the isometry term.
            kl_weight_scale: KL warmup multiplier (architectural, not a tuned
                weight) applied on top of ``w_kl``.
            behavior_pred, behavior_target: predicted/target behavior probs.
            decoded_path: decoded geodesic activations (P, ambient) for the
                geodesic term; skipped if None.
            patched_behavior: behavior probs from a frozen patched model
                (computed by the analysis layer) for the patch term.
            extra_loss: optional pre-computed scalar added verbatim (e.g. a
                router-entropy regularizer); not weighted here.
        """
        device = h.device
        total = torch.zeros((), device=device, dtype=h.dtype)
        metrics: Dict[str, float] = {}

        if self.w_recon != 0.0:
            recon = self.recon_loss(h_hat, h)
            total = total + self.w_recon * recon
            metrics["recon"] = recon.item()

        if self.w_kl != 0.0:
            kl = self.kl_loss(mu, logvar)
            total = total + self.w_kl * kl_weight_scale * kl
            metrics["kl"] = kl.item()

        if self.w_behavior != 0.0:
            if behavior_pred is None or behavior_target is None:
                raise ValueError(
                    "behavior term enabled (w_behavior != 0) but behavior_pred/"
                    "behavior_target not supplied"
                )
            beh = self.behavior_loss(behavior_pred, behavior_target)
            total = total + self.w_behavior * beh
            metrics["behavior"] = beh.item()

        if self.w_isometry != 0.0:
            if u is None:
                raise ValueError(
                    "isometry term enabled (w_isometry != 0) but u not supplied"
                )
            uses_geometry = (
                (
                    geometry_distance in ("cyclic", "ordinal")
                    and geometry_coords is not None
                )
                or geometry_distance == "activation"
                or geometry_distance == "precomputed"
            )
            if not uses_geometry and behavior_target is None:
                raise ValueError(
                    "isometry term enabled (w_isometry != 0) but behavior_target "
                    "not supplied for euclidean geometry"
                )
            if geometry_distance == "precomputed" and (
                geometry_coords is None or geometry_matrix is None
            ):
                raise ValueError(
                    "isometry term with geometry_distance='precomputed' "
                    "requires geometry_coords and geometry_matrix"
                )
            iso = self.isometry_loss(
                u,
                behavior_target,
                geometry_coords=geometry_coords,
                geometry_distance=geometry_distance,
                geometry_period=geometry_period,
                activation_targets=h,
                geometry_matrix=geometry_matrix,
            )
            total = total + self.w_isometry * iso
            metrics["isometry"] = iso.item()

        if self.w_geodesic != 0.0 and decoded_path is not None:
            geo = self.geodesic_loss(decoded_path)
            total = total + self.w_geodesic * geo
            metrics["geodesic"] = geo.item()

        if self.w_patch != 0.0 and patched_behavior is not None:
            if behavior_target is None:
                raise ValueError(
                    "patch term enabled (w_patch != 0) but behavior_target not "
                    "supplied"
                )
            patch = self.behavior_loss(patched_behavior, behavior_target)
            total = total + self.w_patch * patch
            metrics["patch"] = patch.item()

        if self.w_contrastive != 0.0:
            if u is None or class_idx is None:
                raise ValueError(
                    "contrastive term enabled (w_contrastive != 0) but u/class_idx "
                    "not supplied"
                )
            con = self.contrastive_loss(
                u,
                class_idx,
                period=geometry_period,
                margin=contrastive_margin,
                cyclic=(geometry_distance == "cyclic"),
                dist_matrix=(
                    geometry_matrix
                    if geometry_distance == "precomputed"
                    else None
                ),
            )
            total = total + self.w_contrastive * con
            metrics["contrastive"] = con.item()

        if extra_loss is not None:
            total = total + extra_loss

        metrics["total"] = total.item()
        return total, metrics
