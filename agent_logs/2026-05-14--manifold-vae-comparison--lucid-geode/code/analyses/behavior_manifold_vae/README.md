# Behavior Manifold VAE

Behavior Manifold VAE answers: *Can a behavior-aligned VAE learn an activation manifold whose geometry matches the spline/flow manifolds while staying faithful to model behavior?* It trains one VAE arm on the cached subspace features at a single (layer, token) site, optionally aligning the latent geometry to the baseline per-class output distributions, and writes manifold-geometry metrics with names aligned to the spline `path_steering` outputs. The artifacts produced here are read by `compare_manifold_architectures`.

## Overview

```
cached subspace features ──┐
baseline per-class dists ──┼─> train_behavior_aligned_vae ──> VAEManifold + behavior head
output_manifold (belief) ──┘                                        │
                                                                    ├─ encode -> per-class latent centroids U
                                                                    ├─ decoded geodesics (U[i],U[j]) -> D_X (arc length)
   belief control points -> D_Y (Hellinger arc length) ────────────┤
                                                                    └─ compute_isometry_metrics(D_X, D_Y) -> pearson_r
metrics.json / comparison_ready.json / latents.safetensors / ckpt_final.{safetensors,meta.json}
```

When `patch_eval=false` (the debug path) no model weights are loaded — the arm runs on cached features, baseline distributions, and the belief manifold alone.

## Configuration

**Root config** (`causalab/configs/config.yaml`) — shared params this analysis reads:
- `experiment_root` — output root (session-local default under `agent_logs/<session>/artifacts/...`).
- `seed` — dataset generation + training seed.
- `task.*` — `name`, `n_train`, `n_test`, `enumerate_all`, `balanced`, `resample_variable`, `target_variable` (dataset construction, invariant 12).
- `model.name` — recorded in metadata only (no weights loaded when `patch_eval=false`).

**Module config** (`${SESSION_DIR}/code/configs/analysis/behavior_manifold_vae.yaml`):

```yaml
behavior_manifold_vae:
  _name_: behavior_manifold_vae
  _subdir: ${.method}_topo-${.topology}_metric-${.metric}_charts${.n_charts}
  _output_dir: ${experiment_root}/behavior_manifold_vae/${._subdir}
  method: flat_vae            # flat_vae | metric_vae | atlas_vae — VAE architecture arm
  latent_dim: 2               # intrinsic coordinate dimension
  topology: unstructured      # unstructured | s1 | interval | r2 | cylinder — latent topology
  n_charts: 1                 # atlas charts (ignored for non-atlas arms)
  metric: latent_linear       # latent_linear | decoder_pullback | behavior_pullback — geodesic metric for D_X
  loss_weights:               # training loss term weights
    w_recon: 1.0              #   reconstruction
    w_kl: 0.1                 #   KL to prior
    w_behavior: 0.0           #   behavior-prediction term (>0 => behavior_aligned loss set)
    w_isometry: 0.0           #   isometry regularizer (>0 => behavior_aligned loss set)
    w_geodesic: 0.0           #   geodesic-length regularizer
    w_patch: 0.0              #   patch-consistency term
  behavior_distance: hellinger # kl | hellinger | js — behavior-space distance
  activation_site: single_final_token  # single site now; trajectory later
  patch_eval: false           # when true, run the (expensive) frozen-model patch validation
  subspace: null              # explicit subspace subdir, else first discovered
  layers: null                # explicit layer, else subspace best_cell
  token_positions: [last_token]  # explicit token position, else subspace best_cell
  hidden_dims: [256, 256]     # encoder/decoder MLP widths
  behavior_hidden_dims: [128] # behavior-head MLP widths
  lr: 0.001                   # Adam learning rate
  epochs: 200                 # training epochs
  batch_size: 128             # training batch size
  kl_warmup_epochs: 20        # linear KL warmup span
  n_arc_steps: 64             # discretization steps for arc-length integration
  geodesic:                   # solver settings for non-linear metrics
    n_points: 16              #   path points (endpoints included)
    n_iters: 100              #   optimizer iterations
    lr: 0.05                  #   optimizer learning rate
  device: cpu                 # training device
  visualization:
    figure_format: pdf
```

## Outputs

Rooted at `${experiment_root}/behavior_manifold_vae/${subspace}/L{layer}_{token_position}/{method}_topo-{topology}_metric-{metric}_charts{n_charts}_seed{seed}/` (plus a `target_variable` subdir when set).

### Interpretation

- **`metrics.json`** — `reconstruction` and `kl` (lower = better fit), `behavior_distance` (lower = more behavior-faithful; null when baseline distributions are absent), `isometry_pearson_r` (higher = activation geometry matches the belief manifold; null when the belief manifold or a second class is missing), `geodesic_naturalness` (mean distance of decoded geodesic points to the nearest training feature; lower = paths stay on-manifold), `patch_consistency` (null in the debug pass). A `notes` block records any skipped metric and why.
- **`comparison_ready.json`** — the same metrics plus arm descriptors (architecture, task, topology, n_charts, metric, loss_set, layer, token_position, seed) under names aligned across arms; this is the row `compare_manifold_architectures` ingests.

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `metrics.json` | metric dict (+ `notes`) | human reference |
| `comparison_ready.json` | flat arm-descriptor + metric dict | `compare_manifold_architectures` |
| `latents.safetensors` | `{latents: (N, latent_dim)}` | downstream viz / reuse |
| `ckpt_final.safetensors` + `ckpt_final.meta.json` | `vae_state.*`, `mean`, `std` tensors + `config`/`eps` meta | reload via `VAEManifold.from_state_dict` |
| `metadata.json` | run config snapshot (method, topology, metric, ckpt_format, …) | provenance |
| `experiment_metadata.json` | resolved cfg snapshot | provenance |
