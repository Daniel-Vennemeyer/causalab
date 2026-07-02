"""behavior_manifold_vae: train one behavior-aligned VAE arm and write
comparison-ready manifold-geometry metrics.

Trains a VAE-based activation manifold on cached subspace features so it can be
compared head-to-head against the spline ``activation_manifold`` / ``path_steering``
artifacts. The arm runs WITHOUT loading model weights when ``patch_eval=false``
(the debug path): it only needs cached subspace features, the baseline per-class
output distributions, and the ``output_manifold`` belief manifold.

This is a session-local *analysis* (research-question wrapper) — see ARCHITECTURE.md §3.
Layering rules respected by this module:
  - depends on causalab/{neural,methods,io,causal,tasks,runner.helpers} and the
    session-local ``methods.behavior_aligned_vae``; the only peer-analysis imports
    are the explicitly-allowed public loaders (io.pipelines, spline.belief_fit,
    scores.isometry)
  - all disk I/O routes through causalab.io.artifacts primitives (invariant 3),
    except reading cached features via ``safetensors.torch.load_file`` (as
    activation_manifold does)
  - no hyperparameter defaults inline — every knob comes from
    ``cfg.behavior_manifold_vae.<knob>`` or ``cfg.task.*`` / ``cfg.seed``
    (invariants 5, 12)
  - ``cfg.experiment_root`` is the single source of truth for output paths
    (invariant 7)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from causalab.io.artifacts import (
    load_tensor_results,
    save_experiment_metadata,
    save_json_results,
    save_tensor_results,
    save_tensors_with_meta,
)
from causalab.io.counterfactuals import load_counterfactual_examples
from causalab.io.pipelines import find_subspace_dirs, load_subspace_metadata
from causalab.methods.scores.isometry import (
    _decoded_path_length_batched,
    compute_isometry_metrics,
)
from causalab.methods.spline.belief_fit import load_output_manifold
from causalab.runner.helpers import generate_datasets, resolve_task

from methods.behavior_aligned_vae import (  # session-local
    GeodesicSolver,
    compute_metric,
    train_behavior_aligned_vae,
)

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "behavior_manifold_vae"


def _resolve_grid_cell(
    ss_meta: dict,
    explicit_layer: int | None,
    explicit_token_position: str | None,
) -> tuple[int | None, str | None]:
    """Resolve a single (layer, token_position) from grid subspace metadata.

    Mirrors ``activation_manifold._resolve_grid_cell``: explicit config values
    win, then fall back to ``best_cell`` in the subspace metadata.
    """
    best_cell = ss_meta.get("best_cell") or {}
    layer = explicit_layer if explicit_layer is not None else best_cell.get("layer")
    token_position = (
        explicit_token_position
        if explicit_token_position is not None
        else best_cell.get("token_position")
    )
    if layer is not None:
        layer = int(layer)
    if token_position is not None:
        token_position = str(token_position)
    return layer, token_position


def _per_class_centroids(
    u_all: torch.Tensor,
    cls_idx: torch.Tensor,
    n_classes: int,
) -> tuple[torch.Tensor, list[int]]:
    """Mean intrinsic coordinate per class. Returns (U, present_classes).

    Skips classes with no examples; ``present_classes`` lists the kept class
    indices in row order of ``U``.
    """
    rows: list[torch.Tensor] = []
    present: list[int] = []
    for c in range(n_classes):
        mask = cls_idx == c
        if bool(mask.any()):
            rows.append(u_all[mask].mean(dim=0))
            present.append(c)
    if not rows:
        return u_all.new_zeros((0, u_all.shape[1])), present
    return torch.stack(rows, dim=0), present


def _classify_topology(adj: np.ndarray) -> dict[str, Any]:
    """Classify the discovered transition graph's topology from its adjacency.

    Returns a dict with ``kind`` in {cycle, line, complex, irregular, degenerate}
    and ``is_convex`` (True only for a line/interval). The distinction drives the
    topology-adaptive loss gate: curvature-inducing losses (``w_manifold``) help on
    NON-convex manifolds (cycle/2-D) but hurt on a convex line, where straight-line
    latent interpolation is already optimal (confirmed on weekdays/months vs
    alphabet). Robust to spurious edges only if ``adj`` is already denoised.
    """
    deg = adj.sum(axis=1)
    present = deg > 0
    n_nodes = int(present.sum())
    if n_nodes < 3:
        return {"kind": "degenerate", "is_convex": True, "n_nodes": n_nodes}
    sub = adj[present][:, present]
    d = sub.sum(axis=1)
    n_deg1 = int((d == 1).sum())
    max_deg = int(d.max())
    n_edges = int(sub.sum() // 2)
    from scipy.sparse.csgraph import connected_components

    n_comp, _ = connected_components(sub, directed=False)
    if max_deg >= 3:
        kind, is_convex = "complex", False  # 2-D lattice / branching (grid, cylinder)
    elif n_deg1 == 0 and n_edges == n_nodes and n_comp == 1:
        kind, is_convex = "cycle", False  # single loop (weekdays, months)
    elif n_deg1 == 2 and n_edges == n_nodes - 1 and n_comp == 1:
        kind, is_convex = "line", True  # path / interval (alphabet, age)
    else:
        kind, is_convex = "irregular", (n_deg1 >= 2 and max_deg <= 2)
    return {
        "kind": kind,
        "is_convex": is_convex,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_deg1": n_deg1,
        "max_deg": max_deg,
        "n_components": int(n_comp),
    }


def _build_transition_dy(
    task: Any,
    train_dataset: list,
    per_example_dists: torch.Tensor,
    W: int,
    transition_variable: str,
    hold_fixed_vars: list[str],
    edge_min_frac: float = 0.0,
) -> tuple[torch.Tensor | None, np.ndarray | None]:
    """Build a (W, W) relational ``d_y`` from the MODEL's behavioral transitions.

    No topology/period/ordering is injected. We only assume ``transition_variable``
    is a discrete ordinal input we can step through by +1. The cyclicity (if any)
    emerges from how the model's *predicted result class* changes under that step.

    Algorithm:
      1. ``pred_class = per_example_dists[:, :W].argmax(dim=1)`` — the model's
         predicted result class per example (NOT the true label; this is what
         makes the geometry behavior-derived). ``per_example_dists`` is row-aligned
         with ``train_dataset``.
      2. Map each ``transition_variable`` value to its ordinal position via
         ``task.causal_model.values[transition_variable]``.
      3. Group examples by the tuple of ``hold_fixed_vars`` input values. Within a
         group, for every example whose ``pos + 1`` value also appears in the
         group, add an undirected edge between their predicted classes (count in a
         W×W adjacency ``A``).
      4. Treat any ``A > 0`` as an unweighted edge; graph distance = all-pairs
         shortest path. Disconnected pairs -> ``W`` (large finite). Diagonal 0.

    ``edge_min_frac`` (denoising): drop edges whose count is below this fraction of
    the larger endpoint's dominant-edge count. Model misclassifications create rare
    singleton edges (e.g. months: 13 edges for a 12-cycle; alphabet: 26 for a
    21-edge line) that perturb ``d_y`` and can flip the topology classification.
    0.0 = keep every edge (backward-compatible). ~0.3 keeps modal transitions only.

    Returns ``(dist, adj)`` — a (W, W) float tensor of graph distances (or ``None``)
    and the (denoised) 0/1 adjacency ``np.ndarray`` (or ``None``) for topology
    classification.
    """
    M = per_example_dists.shape[0]
    if len(train_dataset) != M:
        raise ValueError(
            "_build_transition_dy: per_example_dists is not row-aligned with "
            f"train_dataset (len(train_dataset)={len(train_dataset)} != "
            f"per_example_dists.shape[0]={M}). They must come from the same "
            "generate_datasets(seed, config) call."
        )

    pred_class = per_example_dists[:, :W].argmax(dim=1)  # (M,) MODEL prediction

    order = list(task.causal_model.values[transition_variable])
    pos = {v: i for i, v in enumerate(order)}

    # Group example indices by the held-fixed input key.
    groups: dict[tuple, dict[int, int]] = {}
    for i, ex in enumerate(train_dataset):
        inp = ex["input"]
        key = tuple(inp[v] for v in hold_fixed_vars)
        p = pos[inp[transition_variable]]
        # Map ordinal position -> predicted class within this group.
        groups.setdefault(key, {})[p] = int(pred_class[i])

    A = np.zeros((W, W), dtype=np.float64)
    n_edges = 0
    for p_to_class in groups.values():
        for p, c in p_to_class.items():
            if (p + 1) in p_to_class:
                c2 = p_to_class[p + 1]
                A[c, c2] += 1
                A[c2, c] += 1
                n_edges += 1

    if n_edges == 0:
        logger.warning(
            "transition d_y found no unit-step pairs; returning None. Ensure "
            "enumerate_all=true and that %r is a discrete ordinal "
            "transition_variable with adjacent values present in the dataset.",
            transition_variable,
        )
        return None, None

    # Denoise: drop edges rare relative to each endpoint's dominant transition.
    if edge_min_frac > 0.0:
        rowmax = A.max(axis=1)  # (W,) dominant-edge count per class
        thresh = edge_min_frac * np.maximum(rowmax[:, None], rowmax[None, :])
        A = np.where(A >= thresh, A, 0.0)

    # Unweighted graph distance (all-pairs shortest path) on the A>0 edges.
    from scipy.sparse.csgraph import shortest_path

    adj = (A > 0).astype(np.float64)
    dist = shortest_path(adj, method="D", directed=False, unweighted=True)
    # Disconnected pairs -> W (large finite); diagonal stays 0.
    dist[~np.isfinite(dist)] = float(W)
    np.fill_diagonal(dist, 0.0)
    return torch.from_numpy(dist).float(), adj


def _build_graph_dy(task: Any, W: int) -> tuple[torch.Tensor | None, np.ndarray | None]:
    """Relational ``d_y`` for graph tasks (graph_walk 2-D): shortest-path distance on
    the KNOWN graph over the W concept nodes.

    Unlike ``_build_transition_dy`` (which steps an ordinal INPUT), the 2-D graph has
    no ordinal to step — so we read the structure directly from the causal model's
    stored graph (``task.causal_model._graph.adjacency``; node ids 0..W-1 align with
    ``intervention_value_index``). This is the SUPERVISED coordinate (the paper uses
    the known graph); discovering it from random-walk transitions is future work.
    Returns ``(dist, adj)`` — the (W, W) graph-distance matrix and the 0/1 adjacency
    (fed to ``_classify_topology`` → 'complex' for a 2-D lattice).
    """
    graph = getattr(task.causal_model, "_graph", None)
    adjacency = getattr(graph, "adjacency", None) if graph is not None else None
    if not adjacency:
        return None, None
    A = np.zeros((W, W), dtype=np.float64)
    for node, nbrs in adjacency.items():
        if int(node) >= W:
            continue
        for n in nbrs:
            if int(n) < W:
                A[int(node), int(n)] = 1.0
                A[int(n), int(node)] = 1.0
    from scipy.sparse.csgraph import shortest_path

    dist = shortest_path(A, method="D", directed=False, unweighted=True)
    dist[~np.isfinite(dist)] = float(W)
    np.fill_diagonal(dist, 0.0)
    return torch.from_numpy(dist).float(), A


def _resample_path(path: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Linearly resample a ``(P, k)`` path to exactly ``num_steps`` points.

    Used to bring a GeodesicSolver path (built with ``geodesic.n_points``
    interior resolution) onto the ``patch_num_steps`` grid the scorers expect.
    """
    if path.shape[0] == num_steps:
        return path
    src = torch.linspace(0.0, 1.0, path.shape[0], dtype=path.dtype)
    dst = torch.linspace(0.0, 1.0, num_steps, dtype=path.dtype)
    cols = []
    for c in range(path.shape[1]):
        # torch has no batched 1-D interp; loop over the few intrinsic dims.
        cols.append(_interp1d(dst, src, path[:, c]))
    return torch.stack(cols, dim=-1)


def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """numpy.interp analogue for monotonically-increasing ``xp``."""
    idx = torch.searchsorted(xp, x).clamp(1, xp.shape[0] - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    w = ((x - x0) / (x1 - x0).clamp(min=1e-12)).clamp(0.0, 1.0)
    return y0 + w * (y1 - y0)


def _plot_latent_coords(
    *,
    u_all: torch.Tensor,
    cls_idx: torch.Tensor,
    U: torch.Tensor,
    present_classes: list[int],
    topology: str,
    periodic_dims: list[int] | None,
    out_dir: str,
    figure_format: str = "png",
) -> str | None:
    """Diagnostic scatter of the learned latent coordinates colored by class.

    The single most informative plot for "did the VAE learn the manifold?":
    per-example intrinsic coords scattered, per-class centroids overlaid and
    connected IN CLASS ORDER (loop closed for cyclic tasks). A clean ring ⇒ the
    latent recovered the cyclic behavioral order; a scrambled/folded loop or
    separated clusters ⇒ it learned separability/density but not the geometry.
    Cyclic 1-D latents (s1) are drawn on the unit circle via (cosθ, sinθ).
    Returns the saved path, or None on failure (never aborts the run).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        u = u_all.detach().cpu().numpy()
        c = cls_idx.detach().cpu().numpy()
        Un = U.detach().cpu().numpy()
        cyclic = bool(periodic_dims) or topology in ("s1", "cylinder")
        cmap = "twilight" if cyclic else "viridis"
        k = u.shape[1]

        def _xy(arr: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
            if k == 1 and cyclic:
                return np.cos(arr[:, 0]), np.sin(arr[:, 0])
            if k == 1:
                return arr[:, 0], np.zeros(arr.shape[0])
            return arr[:, 0], arr[:, 1]

        fig, ax = plt.subplots(figsize=(6, 6))
        xs, ys = _xy(u)
        sc = ax.scatter(xs, ys, c=c, cmap=cmap, s=18, alpha=0.6, zorder=2)
        if Un.shape[0] >= 1:
            cx, cy = _xy(Un)
            order = list(np.argsort(present_classes))
            cxo, cyo = cx[order], cy[order]
            if cyclic and len(order) > 2:
                cxo = np.append(cxo, cxo[0])
                cyo = np.append(cyo, cyo[0])
            ax.plot(cxo, cyo, "-k", lw=1.0, alpha=0.7, zorder=3)
            for idx, cls in enumerate(present_classes):
                ax.scatter([cx[idx]], [cy[idx]], c="k", s=90, marker="*", zorder=4)
                ax.annotate(
                    str(cls), (cx[idx], cy[idx]), fontsize=9, zorder=5,
                    xytext=(3, 3), textcoords="offset points",
                )
        fig.colorbar(sc, ax=ax, label="class index")
        ax.set_title(f"VAE latent ({topology}) — centroids in class order")
        ax.set_aspect("equal", "datalim")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"latent_coords.{figure_format}")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as exc:  # noqa: BLE001 — diagnostic must never abort the run
        logger.warning("latent_coords plot failed: %s", exc)
        return None


def _run_patch_eval(
    *,
    cfg: DictConfig,
    analysis: Any,
    task: Any,
    root: str,
    tv: str | None,
    layer: int,
    token_position: str,
    subspace_out_dir: str,
    ss_method: str,
    ss_meta: dict,
    manifold: Any,
    behavior_fn: Any,
    U: torch.Tensor,
    present_classes: list[int],
    pdims: list[int] | None,
    pers: list[float] | None,
    metrics: dict[str, Any],
    notes: dict[str, str],
    comparison_extra: dict[str, Any],
    features: torch.Tensor,
    project_to_data: bool = False,
    path_builder: Any = None,
) -> None:
    """Decode VAE-steered class-centroid-pair paths into the frozen LM and score
    them on the spline's behavioral axes (coherence + distance_from_behavior_manifold).

    Mirrors ``path_steering``: build the interchange target, compose the
    subspace featurizer with the VAE manifold featurizer onto the unit, build
    ``var_indices``/``eval_samples``, collect per-step output distributions for
    each present class pair, stack into ``(n_pairs, num_steps, n_prompts, W)``,
    and run the two scorers. Writes the patched scores into ``metrics`` and
    ``comparison_extra`` (compare-column keys ``coherence`` /
    ``distance_from_behavior_manifold``, plus ``patch_consistency``).
    """
    import itertools

    from causalab.io.pipelines import load_pipeline
    from causalab.runner.helpers import build_targets_for_grid
    from causalab.analyses.subspace import load_subspace_onto_target
    from causalab.analyses.path_steering.path_mode import _build_geodesic_path
    from causalab.methods.spline.featurizer import ManifoldFeaturizer
    from causalab.methods.metric import tokenize_variable_values
    from causalab.methods.steer.collect import collect_grid_distributions
    from causalab.methods.scores import coherence, distance_from_behavior_manifold
    from causalab.tasks.loader import load_task_counterfactuals

    Wp = U.shape[0]
    if Wp < 2:
        notes["patch_consistency_skipped"] = "fewer than 2 non-empty classes"
        metrics["patch_consistency"] = None
        return

    patch_n_prompts = int(analysis.patch_n_prompts)
    patch_num_steps = int(analysis.patch_num_steps)
    patch_max_pairs = int(analysis.patch_max_pairs)
    patch_batch_size = int(analysis.patch_batch_size)

    # --- Load the frozen model (only now, only when patch_eval) --------------
    pipeline = load_pipeline(
        model_name=cfg.model.name,
        task=task,
        max_new_tokens=cfg.task.max_new_tokens,
        device=cfg.model.get("device", "cuda"),
        dtype=cfg.model.get("dtype"),
        eager_attn=cfg.model.get("eager_attn"),
    )

    try:
        # --- Belief manifold (for distance_from_behavior_manifold) -----------
        om_root = os.path.join(root, "output_manifold")
        bm_sub = None
        if os.path.isdir(om_root):
            for name in sorted(os.listdir(om_root)):
                if os.path.isdir(os.path.join(om_root, name)):
                    bm_sub = name
                    break
        if bm_sub is None:
            notes["patch_consistency_skipped"] = (
                "no output_manifold belief manifold found; cannot score "
                "distance_from_behavior_manifold"
            )
            metrics["patch_consistency"] = None
            return
        if tv:
            bm_sub = os.path.join(bm_sub, tv)
        belief_manifold, _ = load_output_manifold(root, bm_sub)

        # --- Interchange target + composed featurizer (mirror path_steering) -
        # transport (path_builder set) patches ambient PCA points directly via the
        # subspace featurizer (like project_to_data) and needs NO manifold decode,
        # so skip the composed (subspace >> manifold) featurizer entirely.
        k_features = ss_meta.get("k_features")
        interchange_target = None
        if path_builder is None:
            targets, _tp_list = build_targets_for_grid(
                pipeline, task, [layer], position_names=[token_position]
            )
            interchange_target = next(iter(targets.values()))
            load_subspace_onto_target(
                interchange_target, subspace_out_dir, ss_method, k_features
            )
            unit = interchange_target.flatten()[0]
            subspace_feat = unit.featurizer
            # n_features is the PCA-subspace dimensionality (the VAE's ambient dim);
            # the VAE handles its own standardization internally, so no
            # StandardizeFeaturizer stage is added.
            manifold_feat = ManifoldFeaturizer(manifold, n_features=int(k_features))
            composed = subspace_feat >> manifold_feat
            unit.set_featurizer(composed)

        # --- Optional: project-to-data (confirmation test) -------------------
        # Instead of patching the VAE-decoded path point (which may be OFF the
        # realistic-activation distribution), snap each decoded point to the
        # NEAREST real training activation and patch THAT via the subspace
        # featurizer only. If distance-from-behavior-manifold drops toward the
        # spline's, it confirms that off-distribution decoder interpolants — not
        # the latent ordering — are why VAE steering ≈ linear.
        sub_target = None
        feats_ref = features.detach().float().cpu()
        if project_to_data or path_builder is not None:
            targets2, _ = build_targets_for_grid(
                pipeline, task, [layer], position_names=[token_position]
            )
            sub_target = next(iter(targets2.values()))
            load_subspace_onto_target(
                sub_target, subspace_out_dir, ss_method, k_features
            )  # subspace featurizer only (no manifold compose)

        # --- var_indices + eval_samples (mirror path_steering) ---------------
        values = task.intervention_values
        var_indices = tokenize_variable_values(
            pipeline.tokenizer, values, task.result_token_pattern
        )
        cf_mod = load_task_counterfactuals(task.name)
        filtered_samples = cf_mod.generate_dataset(
            task.causal_model, patch_n_prompts, cfg.seed + 100
        )
        eval_samples = filtered_samples[:patch_n_prompts]

        # --- Build + collect per class-centroid pair -------------------------
        use_geodesic = analysis.metric in ("decoder_pullback", "behavior_pullback")
        if use_geodesic:
            solver = GeodesicSolver()

            def metric_fn(u: torch.Tensor) -> torch.Tensor:
                return compute_metric(
                    analysis.metric,
                    u,
                    decode_fn=manifold.decode,
                    behavior_fn=behavior_fn,
                )

        from tqdm import tqdm

        pair_dists_list: list[torch.Tensor] = []
        n_pairs_used = 0
        # Patch a FIXED budget of class-centroid pairs (cost ~constant across
        # tasks). When C(Wp,2) > patch_max_pairs, take an EVEN stride sample, not
        # the first N: combinations() emits all (0,j) before any (1,j), so the
        # first N pairs share class 0 for Wp>patch_max_pairs (e.g. alphabet Wp=22,
        # age Wp=91) — that would measure steering from a single class only. A
        # stride sample spreads pairs across the class set. Reproducible (no RNG).
        _all_pairs = list(itertools.combinations(range(Wp), 2))
        if len(_all_pairs) > patch_max_pairs:
            _stride = len(_all_pairs) // patch_max_pairs
            _patch_pairs = _all_pairs[::_stride][:patch_max_pairs]
        else:
            _patch_pairs = _all_pairs
        for (i, j) in tqdm(_patch_pairs, desc=f"patch[{analysis.metric}]: 8B forwards/pair"):
            if path_builder is not None:
                # transport: integrate the tangent field from real centroid i to j
                # in PCA-subspace coords; patch those ambient points via sub_target.
                patch_grid = path_builder(i, j)
                probs = collect_grid_distributions(
                    pipeline=pipeline,
                    grid_points=patch_grid,
                    interchange_target=sub_target,
                    filtered_samples=eval_samples,
                    var_indices=var_indices,
                    batch_size=patch_batch_size,
                    n_base_samples=patch_n_prompts,
                    average=False,
                    full_vocab_softmax=True,
                )
                pair_dists_list.append(probs)
                n_pairs_used += 1
                continue
            if use_geodesic:
                path = solver.geodesic(
                    U[i],
                    U[j],
                    metric_fn=metric_fn,
                    n_points=int(analysis.geodesic.n_points),
                    n_iters=int(analysis.geodesic.n_iters),
                    lr=float(analysis.geodesic.lr),
                    periodic_dims=pdims,
                    periods=pers,
                )
                grid_points = _resample_path(path, patch_num_steps)
            else:
                # latent_linear: straight line in intrinsic space with periodic
                # shortest-arc wrap (manifold carries periodic_dims/periods).
                grid_points = _build_geodesic_path(
                    U[i], U[j], patch_num_steps, manifold
                )

            # Default: patch the VAE-decoded intrinsic path via the composed
            # (subspace >> manifold) featurizer. project_to_data: decode to PCA,
            # snap each point to the nearest real training activation, and patch
            # that via the subspace featurizer only.
            if project_to_data:
                with torch.no_grad():
                    decoded_pca = manifold.decode(grid_points).detach().cpu()
                    nn = torch.cdist(decoded_pca, feats_ref).argmin(dim=1)
                    patch_grid = feats_ref[nn]  # (steps, k_features) real activations
                patch_target = sub_target
            else:
                patch_grid = grid_points
                patch_target = interchange_target

            # (patch_num_steps, n_prompts, W)
            probs = collect_grid_distributions(
                pipeline=pipeline,
                grid_points=patch_grid,
                interchange_target=patch_target,
                filtered_samples=eval_samples,
                var_indices=var_indices,
                batch_size=patch_batch_size,
                n_base_samples=patch_n_prompts,
                average=False,
                full_vocab_softmax=True,
            )
            pair_dists_list.append(probs)
            n_pairs_used += 1

        # (n_pairs, patch_num_steps, n_prompts, W)
        pair_distributions = torch.stack(pair_dists_list)

        # --- Score on the spline's behavioral axes ---------------------------
        coh = coherence.compute_score(pair_distributions)
        dist = distance_from_behavior_manifold.compute_score(
            pair_distributions, belief_manifold=belief_manifold
        )
        coherence_patched = float(coh["mean"])
        distance_patched = float(dist["mean"])

        # Same compare-column keys as the spline arm, plus patch_consistency.
        metrics["coherence"] = coherence_patched
        metrics["distance_from_behavior_manifold"] = distance_patched
        metrics["patch_consistency"] = coherence_patched
        comparison_extra["coherence"] = coherence_patched
        comparison_extra["distance_from_behavior_manifold"] = distance_patched
        comparison_extra["patch_consistency"] = coherence_patched
        notes["patch_consistency"] = (
            f"patched VAE-decoded paths; patch_n_prompts={patch_n_prompts}, "
            f"patch_num_steps={patch_num_steps}, n_pairs={n_pairs_used}"
        )
    finally:
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the behavior_manifold_vae analysis (single VAE arm)."""
    # Hard-cap CPU threads in-process when the sweep requests it. Env vars alone
    # do NOT reliably constrain this torch build (observed workers pinning ~30
    # cores each → node thrash when run in parallel). torch.set_num_threads is
    # authoritative, so re-apply OMP_NUM_THREADS here. Unset (single one-off
    # run) → leave torch's default so it can use the box.
    _omp = os.environ.get("OMP_NUM_THREADS")
    if _omp:
        _n = max(1, int(_omp))
        try:
            torch.set_num_threads(_n)
        except Exception:  # noqa: BLE001
            pass
        try:
            torch.set_num_interop_threads(_n)
        except Exception:  # noqa: BLE001 — only settable before parallel work
            pass

    analysis = cfg[ANALYSIS_NAME]
    root = cfg.experiment_root
    tv = cfg.task.get("target_variable")
    device = analysis.device

    # --- Task + datasets (sizes from cfg.task, seed from cfg.seed) -----------
    task, _task_cfg = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=tv,
        seed=cfg.seed,
    )
    train_dataset, _test_dataset = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        balanced=cfg.task.get("balanced", False),
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
    )

    # --- Subspace discovery (single-arm: use the first subspace dir) ---------
    if analysis.subspace is not None:
        subspace_subs = [analysis.subspace]
    else:
        subspace_subs = find_subspace_dirs(root)
        if not subspace_subs:
            raise ValueError(
                f"No subspace directories found in {root}/subspace/. "
                "Run the subspace analysis first."
            )
    ss_sub = subspace_subs[0]
    ss_meta = load_subspace_metadata(root, ss_sub, target_variable=tv)
    ss_method = ss_meta.get("method", "pca")
    ss_mode = ss_meta.get("mode", "single")

    explicit_layers = analysis.get("layers", None)
    explicit_token_positions = analysis.get("token_positions", None)
    explicit_layer = int(explicit_layers[0]) if explicit_layers else None
    explicit_token_position = (
        str(explicit_token_positions[0]) if explicit_token_positions else None
    )

    if ss_mode == "grid":
        layer, token_position = _resolve_grid_cell(
            ss_meta, explicit_layer, explicit_token_position
        )
        if layer is None or token_position is None:
            raise ValueError(
                f"Cannot resolve grid cell for subspace {ss_sub}: set "
                "behavior_manifold_vae.layers and .token_positions, or ensure "
                "subspace metadata contains best_cell."
            )
    else:
        layer = ss_meta.get("layer")
        token_position = ss_meta.get("token_position") or explicit_token_position
        if layer is None:
            raise ValueError(f"Missing layer in subspace metadata for {ss_sub}.")
        layer = int(layer)

    # --- Resolve cell_dir (mirror activation_manifold) -----------------------
    subspace_out_dir = os.path.join(root, "subspace", ss_sub)
    if tv:
        subspace_out_dir = os.path.join(subspace_out_dir, tv)
    if ss_mode == "grid" and ss_method == "pca":
        cell_dir = os.path.join(
            subspace_out_dir, "layer_x_pos", f"L{layer}_{token_position}"
        )
    else:
        cell_dir = subspace_out_dir

    # --- Load cached features (row-aligned with the saved subspace dataset) --
    from safetensors.torch import load_file

    features_path = os.path.join(cell_dir, "features", "training_features.safetensors")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Cached features not found at {features_path}. "
            "behavior_manifold_vae runs on cached subspace features; re-run the "
            "subspace analysis to populate the cache first."
        )
    features = load_file(features_path)["features"].float()

    saved_ds_path = os.path.join(cell_dir, "train_dataset.json")
    if not os.path.exists(saved_ds_path):
        saved_ds_path = os.path.join(subspace_out_dir, "train_dataset.json")
    if os.path.exists(saved_ds_path):
        train_dataset = load_counterfactual_examples(saved_ds_path, task.causal_model)
        logger.info("Loaded saved subspace dataset: %d examples", len(train_dataset))

    # --- Behavior targets from baseline per-class output distributions -------
    classes = task.intervention_values
    W = len(classes)
    n_behavior = W

    cls_idx = torch.tensor(
        [task.intervention_value_index(ex) for ex in train_dataset], dtype=torch.long
    )

    behavior_targets: torch.Tensor | None = None
    behavior_targets_available = False
    # Pass EVERY weight in the config block (not a hardcoded subset) so new terms
    # — w_centroid_iso, w_compactness, w_manifold, … — actually reach training.
    # (A previous hardcoded list silently dropped them, defaulting them to 0.)
    loss_weights = {
        str(k): float(v)
        for k, v in OmegaConf.to_container(
            analysis.loss_weights, resolve=True
        ).items()
    }

    # --- Relational-geometry knobs (defaults preserve euclidean behavior) ----
    behavior_geometry = str(analysis.get("behavior_geometry", "euclidean"))
    behavior_period_cfg = analysis.get("behavior_period", None)
    geometry_period = (
        float(behavior_period_cfg) if behavior_period_cfg is not None else float(W)
    )
    contrastive_margin = float(analysis.get("contrastive_margin", 1.0))
    train_on_centroids = bool(analysis.get("train_on_centroids", False))
    # Per-example ordinal class positions; double as contrastive class_idx.
    geometry_coords = cls_idx.float()

    # --- Transition-derived d_y (behavior_geometry == "precomputed") ---------
    # Build a (W, W) relational matrix from the MODEL's behavioral transitions:
    # step the ordinal input ``transition_variable`` by +1 (entity/other inputs
    # held fixed), read how the model's predicted result class moves, and use
    # graph distance on that transition graph as d_y. The ring (if any) emerges
    # from the model's +1 behavior — no period/ordering/topology is declared.
    transition_variable = str(analysis.get("transition_variable", "number"))
    transition_hold_fixed_cfg = analysis.get("transition_hold_fixed", None)
    geometry_matrix: torch.Tensor | None = None
    transition_edge_count: int | None = None
    transition_hold_fixed: list[str] | None = None
    discovered_topology: dict[str, Any] | None = None
    transition_adj = None
    if behavior_geometry == "graph":
        # Known-graph d_y (graph_walk 2-D): shortest-path distance on the task graph.
        # No ordinal input to step; read the structure from the stored causal graph.
        geometry_matrix, transition_adj = _build_graph_dy(task, W)
        if geometry_matrix is None:
            raise ValueError(
                "behavior_geometry=graph requires a task with a stored graph "
                "(task.causal_model._graph.adjacency), e.g. graph_walk."
            )
        transition_edge_count = int((geometry_matrix.numpy() == 1.0).sum() // 2)
        discovered_topology = _classify_topology(transition_adj)
        logger.info(
            "graph d_y built: W=%d, edges=%d, discovered_topology=%s",
            W, transition_edge_count, discovered_topology,
        )
        if bool(analysis.get("adapt_losses_to_topology", False)) and discovered_topology.get("is_convex", False):
            for wk in ("w_manifold", "w_geodesic"):
                if loss_weights.get(wk, 0.0) != 0.0:
                    logger.info("topology=%s (convex) -> gating %s -> 0.0", discovered_topology["kind"], wk)
                    loss_weights[wk] = 0.0
    elif behavior_geometry == "precomputed":
        # Input variable names come from the causal model (exogenous vars with no
        # parents — e.g. entity, number). NOTE: ``ex["input"]`` is a CausalTrace,
        # not a dict — it supports ``trace[var]`` lookup but has no ``.keys()``,
        # so we enumerate via ``task.causal_model.inputs``.
        input_var_names = [str(v) for v in task.causal_model.inputs]
        if transition_variable not in input_var_names:
            raise ValueError(
                f"transition_variable={transition_variable!r} is not an input "
                f"variable of the task (inputs: {input_var_names}). The transition "
                "d_y needs a discrete ordinal INPUT to step through."
            )
        if transition_hold_fixed_cfg is not None:
            transition_hold_fixed = [str(v) for v in transition_hold_fixed_cfg]
        else:
            transition_hold_fixed = [
                v for v in input_var_names if v != transition_variable
            ]

        om_root = os.path.join(root, "output_manifold")
        per_example_dists = load_tensor_results(
            om_root, "per_example_output_dists.safetensors"
        )["dists"].float()
        edge_min_frac = float(analysis.get("transition_edge_min_frac", 0.0))
        geometry_matrix, transition_adj = _build_transition_dy(
            task=task,
            train_dataset=train_dataset,
            per_example_dists=per_example_dists,
            W=W,
            transition_variable=transition_variable,
            hold_fixed_vars=transition_hold_fixed,
            edge_min_frac=edge_min_frac,
        )
        if geometry_matrix is None:
            raise ValueError(
                "transition d_y needs unit-step pairs; ensure enumerate_all and a "
                f"discrete ordinal transition_variable (got {transition_variable!r}, "
                f"hold_fixed={transition_hold_fixed}). No edges were found."
            )
        transition_edge_count = int((geometry_matrix.numpy() == 1.0).sum() // 2)
        logger.info(
            "transition d_y built: W=%d, transition_variable=%s, hold_fixed=%s, "
            "adjacency edges (graph-dist==1 pairs)=%d, edge_min_frac=%.2f",
            W,
            transition_variable,
            transition_hold_fixed,
            transition_edge_count,
            edge_min_frac,
        )

        # --- Topology-adaptive loss gate -------------------------------------
        # Classify the DISCOVERED graph (cycle / line / 2-D) and, if enabled, gate
        # curvature-inducing losses: w_manifold HELPS on non-convex manifolds
        # (cycle/2-D) but HURTS on a convex line, where straight-line latent
        # interpolation is already optimal. This is data-derived (from behavioral
        # transitions), not hand-injected, and lets ONE recipe be optimal across
        # topologies (weekdays/months cyclic vs alphabet/age linear).
        discovered_topology = _classify_topology(transition_adj)
        logger.info("discovered topology: %s", discovered_topology)
        if bool(analysis.get("adapt_losses_to_topology", False)):
            if discovered_topology.get("is_convex", False):
                for wk in ("w_manifold", "w_geodesic"):
                    if loss_weights.get(wk, 0.0) != 0.0:
                        logger.info(
                            "topology=%s (convex) -> gating %s %.3f -> 0.0",
                            discovered_topology["kind"],
                            wk,
                            loss_weights[wk],
                        )
                        loss_weights[wk] = 0.0

    # ======================================================================
    # TRANSPORT ARM: no VAE, no decoder, no parametric manifold. Learn a tangent
    # field on activation space and steer by integrating it from real centroids.
    # Self-contained (early return); writes comparison_ready.json under the same
    # behavior_manifold_vae tree so compare_architectures aggregates it.
    # ======================================================================
    if str(analysis.method) == "transport":
        from methods.behavioral_transport import (
            discovered_order,
            train_transport,
            integrate_path,
        )

        # Coordinate for transport: DISCOVERED (transition graph) by default, or the
        # task's GROUND-TRUTH ordinal (paper's coordinate) as an upper-bound that
        # isolates the transport MECHANISM from discovery noise. Use ground-truth
        # only where the discovered graph is too noisy for a 1-D chain (e.g. alphabet:
        # sparse 2-increment stepping + model errors give a hubbed, max_deg-7 graph).
        if behavior_geometry == "graph":
            # 2-D graph transport: coords = KNOWN node coordinates, neighbors = graph
            # edges, per-dim periodicity from the graph. Field learns a (D x d) Jacobian
            # (Delta_h = J @ dz) -> factored per-axis control.
            _graph = getattr(task.causal_model, "_graph", None)
            if _graph is None or transition_adj is None:
                raise ValueError("graph transport requires a task graph + behavior_geometry=graph.")
            present_classes = [c for c in range(W) if bool((cls_idx == c).any())]
            coords_full = np.array(
                [list(_graph.coordinates[c]) for c in range(W)], dtype=np.float64
            )  # (W, d)
            periods = np.zeros(coords_full.shape[1])
            for _dim, _p in (getattr(_graph, "periodic_dims", {}) or {}).items():
                if int(_dim) < periods.shape[0]:
                    periods[int(_dim)] = float(_p)
            periodic = bool((periods > 0).any())
            n_ranks = len(present_classes)
            U_amb = torch.stack(
                [features[cls_idx == c].mean(dim=0) for c in present_classes], dim=0
            )
            logger.info(
                "graph transport: d=%d periods=%s present=%d",
                coords_full.shape[1], periods.tolist(), n_ranks,
            )
            field, tinfo = train_transport(
                features=features, cls_idx=cls_idx, ranks=coords_full, periodic=False, W=W,
                adjacency=transition_adj, periods=periods,
                hidden_dims=list(analysis.hidden_dims), epochs=int(analysis.epochs),
                lr=float(analysis.lr), w_density=float(analysis.get("transport_w_density", 0.0)),
                seed=int(cfg.seed), device=device,
            )
            logger.info("transport trained: %s", tinfo)
            _cf = coords_full

            def _transport_path(i: int, j: int) -> torch.Tensor:
                return integrate_path(
                    field, U_amb[i].to(device), _cf[present_classes[i]], _cf[present_classes[j]],
                    n_steps=int(analysis.patch_num_steps), periods=periods,
                ).detach().cpu().float()
        else:
            _use_gt = bool(analysis.get("transport_use_ground_truth_coord", False))
            _coord_source = "ground_truth" if _use_gt else "discovered"
            present_classes = [c for c in range(W) if bool((cls_idx == c).any())]
            if _use_gt:
                _emb = (task.causal_model.embeddings or {}).get("result")
                _vals = task.intervention_values
                _periods = task.causal_model.periods or {}
                periodic = "result" in _periods
                _coords = {
                    c: (float(_emb(_vals[c])[0]) if _emb else float(c)) for c in present_classes
                }
                present_classes.sort(key=lambda c: _coords[c])
                rank = np.full(W, -1, dtype=np.int64)
                for pos, c in enumerate(present_classes):
                    rank[c] = pos
            else:
                if transition_adj is None:
                    raise ValueError("transport requires behavior_geometry=precomputed (transition graph).")
                _ord = discovered_order(transition_adj)
                if _ord is None:
                    raise ValueError(
                        "transport: discovered graph is not a 1-D chain (branched/2-D); "
                        "set transport_use_ground_truth_coord=true to test the mechanism, "
                        "or use the atlas (future work). "
                        f"discovered_topology={discovered_topology}"
                    )
                rank, periodic = _ord
                present_classes = [c for c in present_classes if rank[c] >= 0]
                present_classes.sort(key=lambda c: int(rank[c]))
            n_ranks = len(present_classes)
            logger.info("transport coordinate: source=%s periodic=%s n_ranks=%d", _coord_source, periodic, n_ranks)
            U_amb = torch.stack(
                [features[cls_idx == c].mean(dim=0) for c in present_classes], dim=0
            )  # (Wp, k) real per-class centroids
            z_present = np.array([float(rank[c]) for c in present_classes])

            field, tinfo = train_transport(
                features=features,
                cls_idx=cls_idx,
                ranks=rank,
                periodic=periodic,
                W=W,
                hidden_dims=list(analysis.hidden_dims),
                epochs=int(analysis.epochs),
                lr=float(analysis.lr),
                neighbor_radius=int(analysis.get("transport_neighbor_radius", 1)),
                w_density=float(analysis.get("transport_w_density", 0.0)),
                seed=int(cfg.seed),
                device=device,
            )
            logger.info("transport trained: %s (periodic=%s, n_ranks=%d)", tinfo, periodic, n_ranks)

            def _transport_path(i: int, j: int) -> torch.Tensor:
                return integrate_path(
                    field,
                    U_amb[i].to(device),
                    float(z_present[i]),
                    float(z_present[j]),
                    n_steps=int(analysis.patch_num_steps),
                    periodic=periodic,
                    period=float(n_ranks),
                ).detach().cpu().float()

        metrics_t: dict[str, Any] = {
            "reconstruction": None,
            "kl": None,
            "behavior_distance": None,
            "isometry_pearson_r": None,
            "geodesic_naturalness": None,
            "patch_consistency": None,
            "final_transport_loss": tinfo.get("final_loss"),
        }
        notes_t: dict[str, str] = {}
        extra_t: dict[str, Any] = {}

        # --- isometry: transport-path arc length (D_X) vs belief geodesic (D_Y) --
        try:
            Wp = U_amb.shape[0]
            if Wp >= 2:
                ia, ja = np.triu_indices(Wp, 1)
                D_X = np.zeros((Wp, Wp))
                for a, b in zip(ia.tolist(), ja.tolist()):
                    p = _transport_path(a, b)
                    arc = float((p[1:] - p[:-1]).norm(dim=-1).sum())
                    D_X[a, b] = D_X[b, a] = arc
                om_root = os.path.join(root, "output_manifold")
                bm_sub = next(
                    (n for n in sorted(os.listdir(om_root)) if os.path.isdir(os.path.join(om_root, n))),
                    None,
                ) if os.path.isdir(om_root) else None
                if bm_sub is not None:
                    if tv:
                        bm_sub = os.path.join(bm_sub, tv)
                    belief_manifold, _ = load_output_manifold(root, bm_sub)
                    bel = belief_manifold.control_points
                    if bel.shape[0] == W:
                        pres_t = torch.tensor(present_classes, dtype=torch.long)
                        bp = bel[pres_t].to(torch.float32)
                        bpd = list(getattr(belief_manifold, "periodic_dims", None) or []) or None
                        bper = (list(belief_manifold.periods) if hasattr(belief_manifold, "periods") else []) or None
                        D_Y = np.zeros((Wp, Wp))
                        with torch.no_grad():
                            bl = _decoded_path_length_batched(
                                bp[ia], bp[ja], decode_fn=belief_manifold.decode,
                                n_steps=int(analysis.n_arc_steps), periodic_dims=bpd, periods=bper,
                            ) / (2.0 ** 0.5)
                        bl = bl.cpu().numpy()
                        D_Y[ia, ja] = bl; D_Y[ja, ia] = bl
                        iso = compute_isometry_metrics(D_X, D_Y)
                        metrics_t["isometry_pearson_r"] = float(iso["pearson_r"]) if iso.get("pearson_r") is not None else None
        except Exception as exc:  # isometry is diagnostic; never fatal
            notes_t["isometry_skipped"] = str(exc)

        # --- patch-grounded steering (the decisive metric) ----------------------
        if bool(analysis.patch_eval):
            _run_patch_eval(
                cfg=cfg, analysis=analysis, task=task, root=root, tv=tv,
                layer=layer, token_position=token_position,
                subspace_out_dir=subspace_out_dir, ss_method=ss_method, ss_meta=ss_meta,
                manifold=None, behavior_fn=None, U=U_amb, present_classes=present_classes,
                pdims=None, pers=None, metrics=metrics_t, notes=notes_t,
                comparison_extra=extra_t, features=features, path_builder=_transport_path,
            )

        # --- output dir + comparison_ready (same schema/tree as the VAE arms) ----
        arm_label_t = analysis.get("arm_label", None) or "transport"
        arm_sub_t = (
            f"{analysis.method}_topo-{analysis.topology}"
            f"_metric-{analysis.metric}_charts{analysis.n_charts}"
            f"_loss-behavior_aligned_{arm_label_t}_seed{cfg.seed}"
        )
        out_dir_t = os.path.join(
            root, "behavior_manifold_vae", ss_sub, f"L{layer}_{token_position}", arm_sub_t
        )
        if tv:
            out_dir_t = os.path.join(out_dir_t, tv)
        os.makedirs(out_dir_t, exist_ok=True)
        comparison_ready_t = {
            "architecture": "transport",
            "arm_label": arm_label_t,
            "task": cfg.task.name,
            "topology": analysis.topology,
            "n_charts": analysis.n_charts,
            "metric": "transport",
            "loss_set": "behavior_aligned",
            "layer": layer,
            "token_position": token_position,
            "seed": cfg.seed,
            "reconstruction": None,
            "kl": None,
            "behavior_distance": None,
            "isometry_pearson_r": metrics_t["isometry_pearson_r"],
            "geodesic_naturalness": None,
            "patch_consistency": metrics_t.get("patch_consistency"),
            "behavior_geometry": behavior_geometry,
            "discovered_topology": discovered_topology.get("kind") if discovered_topology else None,
        }
        comparison_ready_t.update(extra_t)
        if notes_t:
            metrics_t["notes"] = notes_t
        save_json_results(metrics_t, out_dir_t, "metrics.json")
        save_json_results(comparison_ready_t, out_dir_t, "comparison_ready.json")
        save_json_results(
            {"analysis": ANALYSIS_NAME, "method": "transport", "transport": tinfo,
             "periodic": bool(periodic), "n_ranks": n_ranks},
            out_dir_t, "metadata.json",
        )
        logger.info("transport arm complete: %s", out_dir_t)
        return {"out_dir": out_dir_t, "metrics": metrics_t, "comparison_ready": comparison_ready_t}

    baseline_dir = os.path.join(root, "baseline")
    baseline_path = os.path.join(baseline_dir, "per_class_output_dists.safetensors")
    if os.path.exists(baseline_path):
        dists = load_tensor_results(baseline_dir, "per_class_output_dists.safetensors")[
            "dists"
        ].float()
        # Concept slice: first W columns, clamp >= 0, renormalize rows.
        target_per_class = dists[:, :W].clamp(min=0.0)
        row_sums = target_per_class.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        target_per_class = target_per_class / row_sums  # (W, W)
        behavior_targets = target_per_class[cls_idx]  # (N, W)
        behavior_targets_available = True
    else:
        logger.warning(
            "Baseline per-class output dists not found at %s; training "
            "reconstruction + KL only (behavior/patch weights -> 0).",
            baseline_path,
        )
        loss_weights["w_behavior"] = 0.0
        loss_weights["w_patch"] = 0.0
        # The precomputed (transition) isometry geometry does NOT depend on the
        # baseline behavior targets — its d_y is the transition matrix — so keep
        # w_isometry. Every other geometry's euclidean d_y needs the targets.
        if behavior_geometry not in ("precomputed", "graph"):
            loss_weights["w_isometry"] = 0.0

    # --- Build the training set ----------------------------------------------
    # Centroid-supervised upper bound: train on per-class mean features (W rows)
    # against per-class behavior targets, with ordinal coords arange(W). No val
    # split (W is tiny). Downstream metrics still encode the FULL feature set, so
    # the module-level ``features``/``cls_idx`` are left untouched.
    if train_on_centroids:
        if not behavior_targets_available:
            raise ValueError(
                "train_on_centroids=true requires baseline per-class output "
                "distributions, but none were found."
            )
        present_t = torch.unique(cls_idx, sorted=True)
        feat_rows = [features[cls_idx == int(c)].mean(dim=0) for c in present_t]
        train_features = torch.stack(feat_rows, dim=0)  # (Wp, ambient)
        train_targets = target_per_class[present_t]  # (Wp, W)
        train_geometry_coords = present_t.float()
        val_features = None
        val_targets = None
        val_geometry_coords = None
        logger.info(
            "centroid-supervised mode ON: training on %d per-class centroid rows",
            train_features.shape[0],
        )
    else:
        # --- Validation split (last 10% of rows) -----------------------------
        n = features.shape[0]
        n_val = max(1, int(round(0.1 * n))) if n >= 10 else 0
        if n_val > 0:
            train_slice = slice(0, n - n_val)
            val_slice = slice(n - n_val, n)
            train_features = features[train_slice]
            val_features = features[val_slice]
            train_targets = (
                behavior_targets[train_slice] if behavior_targets is not None else None
            )
            val_targets = (
                behavior_targets[val_slice] if behavior_targets is not None else None
            )
            train_geometry_coords = geometry_coords[train_slice]
            val_geometry_coords = geometry_coords[val_slice]
        else:
            train_features = features
            val_features = None
            train_targets = behavior_targets
            val_targets = None
            train_geometry_coords = geometry_coords
            val_geometry_coords = None

    # --- Train the VAE arm ---------------------------------------------------
    result = train_behavior_aligned_vae(
        features=train_features,
        behavior_targets=train_targets,
        method=analysis.method,
        latent_dim=analysis.latent_dim,
        hidden_dims=list(analysis.hidden_dims),
        topology=analysis.topology,
        n_charts=analysis.n_charts,
        behavior_hidden_dims=list(analysis.behavior_hidden_dims),
        n_behavior=n_behavior,
        loss_weights=loss_weights,
        behavior_distance=analysis.behavior_distance,
        lr=analysis.lr,
        epochs=analysis.epochs,
        batch_size=analysis.batch_size,
        kl_warmup_epochs=analysis.kl_warmup_epochs,
        device=device,
        seed=cfg.seed,
        val_features=val_features,
        val_behavior_targets=val_targets,
        geometry_coords=train_geometry_coords,
        val_geometry_coords=val_geometry_coords,
        geometry_distance=(
            "precomputed" if behavior_geometry in ("precomputed", "graph") else behavior_geometry
        ),
        geometry_period=geometry_period,
        geometry_matrix=geometry_matrix,
        contrastive_margin=contrastive_margin,
    )
    manifold = result["manifold"]
    behavior_head = result["behavior_head"]
    final_metrics = result["final_metrics"]

    # --- Output directory ----------------------------------------------------
    # Encode loss_set AND an arm_label in the path. Several arms share the same
    # (method, topology, metric, charts, loss_set) tuple but differ in geometry /
    # contrastive / train_on_centroids / loss weights (e.g. aligned_flat vs strong
    # vs cyclic vs centroid_upper) — without a distinguishing label they collide
    # on disk and overwrite each other. arm_label (set per runner config) makes
    # each arm's output dir unique; falls back to a composite of the
    # distinguishing knobs if not set.
    _lw = analysis.loss_weights
    _loss_set = (
        "behavior_aligned"
        if (float(_lw.get("w_behavior", 0.0)) > 0.0 or float(_lw.get("w_isometry", 0.0)) > 0.0)
        else "recon_only"
    )
    arm_label = analysis.get("arm_label", None)
    if not arm_label:
        _parts = [f"geo-{analysis.behavior_geometry}"]
        if analysis.get("train_on_centroids", False):
            _parts.append("cen")
        _wc = float(_lw.get("w_contrastive", 0.0))
        if _wc:
            _parts.append(f"con{_wc:g}")
        arm_label = "_".join(_parts)
    arm_sub = (
        f"{analysis.method}_topo-{analysis.topology}"
        f"_metric-{analysis.metric}_charts{analysis.n_charts}"
        f"_loss-{_loss_set}_{arm_label}_seed{cfg.seed}"
    )
    out_dir = os.path.join(
        root, "behavior_manifold_vae", ss_sub, f"L{layer}_{token_position}", arm_sub
    )
    if tv:
        out_dir = os.path.join(out_dir, tv)
    os.makedirs(out_dir, exist_ok=True)

    # --- Encode all training features ----------------------------------------
    manifold.eval()
    with torch.no_grad():
        u_all, _resid = manifold.encode(features)
    u_all = u_all.detach().cpu()

    # --- Scalar training/val metrics -----------------------------------------
    def _final(key: str) -> float | None:
        val_key = f"val_{key}"
        if val_key in final_metrics:
            return float(final_metrics[val_key])
        if key in final_metrics:
            return float(final_metrics[key])
        return None

    reconstruction = _final("recon")
    kl = _final("kl")
    behavior_distance_val = _final("behavior") if behavior_targets_available else None

    metrics: dict[str, Any] = {
        "reconstruction": reconstruction,
        "kl": kl,
        "behavior_distance": behavior_distance_val,
        "isometry_pearson_r": None,
        "geodesic_naturalness": None,
        "patch_consistency": None,
    }
    notes: dict[str, str] = {}
    # Extra metrics keyed into BOTH metrics.json and comparison_ready.json. The
    # patch_eval branch fills these with the patched behavioral-axis scores so
    # the VAE arm lands in the SAME compare columns as the spline.
    comparison_extra: dict[str, Any] = {}

    pdims = manifold.periodic_dims or None
    pers = manifold.periods or None

    # --- Per-class VAE latent centroids --------------------------------------
    U, present_classes = _per_class_centroids(u_all, cls_idx, W)

    # --- Behavior function for behavior_pullback metric ----------------------
    # The behavior head consumes INTRINSIC coords (in_dim = intrinsic_dim), so
    # the ambient->behavior map re-encodes: behavior(x) = head(encode(x)).
    behavior_fn = None
    if behavior_head is not None:

        def behavior_fn(x_ambient: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
            u_enc, _ = manifold.encode(x_ambient)
            return behavior_head.predict_dist(u_enc)

    # --- Build D_X paths (activation-side geodesic arc lengths) --------------
    # ``paths_decoded`` holds the decoded ambient path per (i, j) pair for reuse
    # in geodesic_naturalness; ``D_X`` is the (W_present, W_present) matrix.
    Wp = U.shape[0]
    D_X = None
    paths_decoded: list[torch.Tensor] = []
    if Wp >= 2:
        ia, ja = np.triu_indices(Wp, 1)
        D_X = np.zeros((Wp, Wp), dtype=np.float64)
        if analysis.metric == "latent_linear":
            with torch.no_grad():
                lengths = _decoded_path_length_batched(
                    U[ia],
                    U[ja],
                    decode_fn=manifold.decode,
                    n_steps=int(analysis.n_arc_steps),
                    periodic_dims=pdims,
                    periods=pers,
                )
            lengths_np = lengths.detach().cpu().numpy()
            D_X[ia, ja] = lengths_np
            D_X[ja, ia] = lengths_np
            # Decoded straight latent paths for naturalness.
            t = torch.linspace(0.0, 1.0, int(analysis.n_arc_steps) + 1)
            for a, b in zip(ia.tolist(), ja.tolist()):
                u_line = U[a].unsqueeze(0) + t.unsqueeze(1) * (U[b] - U[a]).unsqueeze(0)
                with torch.no_grad():
                    paths_decoded.append(manifold.decode(u_line))
        elif analysis.metric in ("decoder_pullback", "behavior_pullback"):
            if analysis.metric == "behavior_pullback" and behavior_fn is None:
                notes["isometry_skipped"] = (
                    "behavior_pullback metric requested but no behavior head "
                    "available (behavior targets missing)"
                )
                D_X = None
            else:
                solver = GeodesicSolver()

                def metric_fn(u: torch.Tensor) -> torch.Tensor:
                    return compute_metric(
                        analysis.metric,
                        u,
                        decode_fn=manifold.decode,
                        behavior_fn=behavior_fn,
                    )

                from tqdm import tqdm

                for a, b in tqdm(
                    list(zip(ia.tolist(), ja.tolist())),
                    desc=f"isometry geodesics [{analysis.metric}]",
                ):
                    path = solver.geodesic(
                        U[a],
                        U[b],
                        metric_fn=metric_fn,
                        n_points=int(analysis.geodesic.n_points),
                        n_iters=int(analysis.geodesic.n_iters),
                        lr=float(analysis.geodesic.lr),
                        periodic_dims=pdims,
                        periods=pers,
                    )
                    with torch.no_grad():
                        decoded = manifold.decode(path)
                    paths_decoded.append(decoded)
                    arc = (decoded[1:] - decoded[:-1]).norm(dim=-1).sum()
                    D_X[a, b] = float(arc)
                    D_X[b, a] = float(arc)
        else:
            raise ValueError(
                f"unknown metric kind {analysis.metric!r}; expected one of "
                "'latent_linear', 'decoder_pullback', 'behavior_pullback'"
            )
    else:
        notes["isometry_skipped"] = "fewer than 2 non-empty classes"

    # --- isometry_pearson_r: D_X (activation) vs D_Y (belief) ----------------
    if D_X is not None:
        om_root = os.path.join(root, "output_manifold")
        bm_sub = None
        if os.path.isdir(om_root):
            for name in sorted(os.listdir(om_root)):
                if os.path.isdir(os.path.join(om_root, name)):
                    bm_sub = name
                    break
        if bm_sub is None:
            notes["isometry_skipped"] = (
                "no output_manifold belief manifold found; cannot build D_Y"
            )
        else:
            # output_manifold nests its checkpoint under {subdir}/{target_variable}/
            # when a target_variable is set (mirrors path_steering's loader).
            if tv:
                bm_sub = os.path.join(bm_sub, tv)
            belief_manifold, _ = load_output_manifold(root, bm_sub)
            bel_cps = belief_manifold.control_points
            present_t = torch.tensor(present_classes, dtype=torch.long)
            if bel_cps.shape[0] != W:
                notes["isometry_skipped"] = (
                    f"belief manifold has {bel_cps.shape[0]} control points but "
                    f"task has {W} classes; cannot align"
                )
            else:
                bel_present = bel_cps[present_t].to(torch.float32)
                bel_pdims = (
                    list(getattr(belief_manifold, "periodic_dims", None) or []) or None
                )
                bel_pers = (
                    list(belief_manifold.periods)
                    if hasattr(belief_manifold, "periods")
                    else []
                ) or None
                ib, jb = np.triu_indices(Wp, 1)
                D_Y = np.zeros((Wp, Wp), dtype=np.float64)
                with torch.no_grad():
                    bel_lengths = _decoded_path_length_batched(
                        bel_present[ib],
                        bel_present[jb],
                        decode_fn=belief_manifold.decode,
                        n_steps=int(analysis.n_arc_steps),
                        periodic_dims=bel_pdims,
                        periods=bel_pers,
                    ) / (2.0**0.5)
                bel_np = bel_lengths.detach().cpu().numpy()
                D_Y[ib, jb] = bel_np
                D_Y[jb, ib] = bel_np
                iso = compute_isometry_metrics(D_X, D_Y)
                metrics["isometry_pearson_r"] = (
                    float(iso["pearson_r"])
                    if np.isfinite(iso["pearson_r"])
                    else None
                )

    # --- geodesic_naturalness: mean off-manifold energy of decoded paths -----
    if paths_decoded:
        ref = features
        if ref.shape[0] > 2000:
            g = torch.Generator().manual_seed(int(cfg.seed))
            sel = torch.randperm(ref.shape[0], generator=g)[:2000]
            ref = ref[sel]
        all_pts = torch.cat(paths_decoded, dim=0)
        # Compute on CPU so this eval metric is device-agnostic: with device=cuda
        # the decoded paths are on GPU while the cached features are on CPU.
        with torch.no_grad():
            dists_to_ref = torch.cdist(all_pts.cpu(), ref.cpu())
            nearest = dists_to_ref.min(dim=1).values
        metrics["geodesic_naturalness"] = float(nearest.mean())
    else:
        notes.setdefault("geodesic_naturalness_skipped", "no decoded paths built")

    # --- patch_consistency: decode steered VAE paths into the frozen LM ------
    # When patch_eval=false (debug path) keep the exact prior behavior: null +
    # note, no model load. When true, decode each class-centroid-pair path
    # (built in intrinsic space) through the composed subspace>>VAE featurizer,
    # patch it at (layer, token_position) into the frozen model, and score the
    # resulting output distributions on the SAME behavioral axes as the spline
    # (coherence + distance_from_behavior_manifold).
    if not analysis.patch_eval:
        metrics["patch_consistency"] = None
        notes["patch_consistency_skipped"] = "patch_eval=false"
    else:
        _run_patch_eval(
            cfg=cfg,
            analysis=analysis,
            task=task,
            root=root,
            tv=tv,
            layer=layer,
            token_position=token_position,
            subspace_out_dir=subspace_out_dir,
            ss_method=ss_method,
            ss_meta=ss_meta,
            manifold=manifold,
            behavior_fn=behavior_fn,
            U=U,
            present_classes=present_classes,
            pdims=pdims,
            pers=pers,
            metrics=metrics,
            notes=notes,
            comparison_extra=comparison_extra,
            features=features,
            project_to_data=bool(analysis.get("patch_project_to_data", False)),
        )

    if notes:
        metrics["notes"] = notes

    # --- Persist checkpoint (plain-dict round-trip via tensors_with_meta) -----
    # state_dict_to_save() returns a dict mixing tensors (vae_state, mean, std)
    # and scalars (config, eps); save_module wants an nn.Module, so we split it
    # into a flat tensor payload + JSON meta. A loader rebuilds the dict and
    # calls VAEManifold.from_state_dict.
    save_state = manifold.state_dict_to_save()
    ckpt_tensors: dict[str, torch.Tensor] = {}
    for k, v in save_state["vae_state"].items():
        ckpt_tensors[f"vae_state.{k}"] = v.cpu()
    ckpt_tensors["mean"] = save_state["mean"].cpu()
    ckpt_tensors["std"] = save_state["std"].cpu()
    ckpt_meta = {
        "config": save_state["config"],
        "eps": save_state["eps"],
        "loader": "VAEManifold.from_state_dict",
        "vae_state_keys": list(save_state["vae_state"].keys()),
    }
    save_tensors_with_meta(ckpt_tensors, ckpt_meta, out_dir, "ckpt_final")
    ckpt_format = "tensors_with_meta:ckpt_final"

    # --- Persist latents -----------------------------------------------------
    save_tensor_results({"latents": u_all}, out_dir, "latents.safetensors")

    # --- Diagnostic: latent-coordinate plot (circle vs scramble vs clusters) -
    _plot_latent_coords(
        u_all=u_all,
        cls_idx=cls_idx,
        U=U,
        present_classes=present_classes,
        topology=str(analysis.topology),
        periodic_dims=list(getattr(manifold, "periodic_dims", []) or []),
        out_dir=out_dir,
        figure_format="png",  # diagnostic — PNG for quick eyeballing across arms/seeds
    )

    # --- comparison_ready descriptor (names aligned across arms) -------------
    w_behavior = loss_weights["w_behavior"]
    w_isometry = loss_weights["w_isometry"]
    loss_set = (
        "behavior_aligned"
        if (w_behavior > 0.0 or w_isometry > 0.0)
        else "recon_only"
    )
    arch_map = {
        "flat_vae": "flat_vae",
        "metric_vae": "metric_vae",
        "atlas_vae": "atlas_vae",
    }
    architecture = arch_map.get(analysis.method, analysis.method)
    comparison_ready = {
        "architecture": architecture,
        "arm_label": arm_label,
        "task": cfg.task.name,
        "topology": analysis.topology,
        "n_charts": analysis.n_charts,
        "metric": analysis.metric,
        "loss_set": loss_set,
        "layer": layer,
        "token_position": token_position,
        "seed": cfg.seed,
        "reconstruction": metrics["reconstruction"],
        "kl": metrics["kl"],
        "behavior_distance": metrics["behavior_distance"],
        "isometry_pearson_r": metrics["isometry_pearson_r"],
        "geodesic_naturalness": metrics["geodesic_naturalness"],
        "patch_consistency": metrics["patch_consistency"],
        "behavior_geometry": behavior_geometry,
        "train_on_centroids": train_on_centroids,
        "w_contrastive": loss_weights["w_contrastive"],
        "transition_variable": (
            transition_variable if behavior_geometry == "precomputed" else None
        ),
        "transition_hold_fixed": (
            transition_hold_fixed if behavior_geometry == "precomputed" else None
        ),
        "transition_edge_count": transition_edge_count,
        "discovered_topology": (
            discovered_topology.get("kind") if discovered_topology else None
        ),
        "w_manifold": loss_weights.get("w_manifold", 0.0),
    }
    # Patched behavioral-axis scores (only present when patch_eval=true). Use the
    # SAME keys as the spline so both arms populate the same compare columns.
    comparison_ready.update(comparison_extra)

    # --- Persist metrics + comparison_ready + metadata -----------------------
    save_json_results(metrics, out_dir, "metrics.json")
    save_json_results(comparison_ready, out_dir, "comparison_ready.json")

    meta = {
        "analysis": ANALYSIS_NAME,
        "method": analysis.method,
        "topology": analysis.topology,
        "n_charts": analysis.n_charts,
        "metric": analysis.metric,
        "latent_dim": analysis.latent_dim,
        "layer": layer,
        "token_position": token_position,
        "subspace": ss_sub,
        "loss_weights": loss_weights,
        "behavior_targets_available": behavior_targets_available,
        "behavior_geometry": behavior_geometry,
        "behavior_period": geometry_period,
        "contrastive_margin": contrastive_margin,
        "train_on_centroids": train_on_centroids,
        "transition_variable": (
            transition_variable if behavior_geometry == "precomputed" else None
        ),
        "transition_hold_fixed": (
            transition_hold_fixed if behavior_geometry == "precomputed" else None
        ),
        "transition_edge_count": transition_edge_count,
        "ckpt_format": ckpt_format,
        "model": cfg.model.name,
        "task": cfg.task.name,
        "seed": cfg.seed,
        "n_present_classes": int(Wp),
    }
    save_json_results(meta, out_dir, "metadata.json")
    save_experiment_metadata(
        OmegaConf.to_container(cfg, resolve=True), out_dir, "experiment_metadata.json"
    )

    logger.info("behavior_manifold_vae arm complete: %s", out_dir)
    return {"out_dir": out_dir, "metrics": metrics, "comparison_ready": comparison_ready}
