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
        targets, _tp_list = build_targets_for_grid(
            pipeline, task, [layer], position_names=[token_position]
        )
        interchange_target = next(iter(targets.values()))

        k_features = ss_meta.get("k_features")
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

        pair_dists_list: list[torch.Tensor] = []
        n_pairs_used = 0
        for (i, j) in itertools.combinations(range(Wp), 2):
            if n_pairs_used >= patch_max_pairs:
                break
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

            # (patch_num_steps, n_prompts, W)
            probs = collect_grid_distributions(
                pipeline=pipeline,
                grid_points=grid_points,
                interchange_target=interchange_target,
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
    loss_weights = {
        "w_recon": float(analysis.loss_weights.w_recon),
        "w_kl": float(analysis.loss_weights.w_kl),
        "w_behavior": float(analysis.loss_weights.w_behavior),
        "w_isometry": float(analysis.loss_weights.w_isometry),
        "w_geodesic": float(analysis.loss_weights.w_geodesic),
        "w_patch": float(analysis.loss_weights.w_patch),
    }

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
            "reconstruction + KL only (behavior/isometry/patch weights -> 0).",
            baseline_path,
        )
        loss_weights["w_behavior"] = 0.0
        loss_weights["w_isometry"] = 0.0
        loss_weights["w_patch"] = 0.0

    # --- Validation split (last 10% of rows) ---------------------------------
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
    else:
        train_features = features
        val_features = None
        train_targets = behavior_targets
        val_targets = None

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
    )
    manifold = result["manifold"]
    behavior_head = result["behavior_head"]
    final_metrics = result["final_metrics"]

    # --- Output directory ----------------------------------------------------
    # Encode loss_set in the path so a recon-only and a behavior-aligned arm with
    # otherwise-identical descriptors (method/topology/metric/charts/seed) do not
    # write to the same directory and clobber each other.
    _lw = analysis.loss_weights
    _loss_set = (
        "behavior_aligned"
        if (float(_lw.get("w_behavior", 0.0)) > 0.0 or float(_lw.get("w_isometry", 0.0)) > 0.0)
        else "recon_only"
    )
    arm_sub = (
        f"{analysis.method}_topo-{analysis.topology}"
        f"_metric-{analysis.metric}_charts{analysis.n_charts}"
        f"_loss-{_loss_set}_seed{cfg.seed}"
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

                for a, b in zip(ia.tolist(), ja.tolist()):
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
        with torch.no_grad():
            dists_to_ref = torch.cdist(all_pts, ref)
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
