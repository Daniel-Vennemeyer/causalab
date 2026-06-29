# Custom Method Spec: `behavior_aligned_vae`

## Purpose

Provide reusable session-local primitives for training and evaluating activation-space VAEs that expose intrinsic coordinates, decoder immersion maps, chart/atlas routing, and differentiable metric estimates.

## Shipped-code relationship

This method is a new prototype, not a replacement for `causalab.methods.spline` or `causalab.methods.flow`. It should live under `${SESSION_DIR}/code/methods/behavior_aligned_vae/` and be imported only by session-local analyses.

## Required components

- `ActivationVAE`: MLP encoder/decoder baseline with Gaussian latent `q_phi(z | h)`, reparameterization, reconstruction loss, and KL loss.
- `AtlasVAE`: chart router `r_phi(k | h)`, per-chart latent heads, and per-chart decoders `f_{theta,k}(z_k)`.
- `BehaviorHead`: lightweight differentiable predictor from latent or decoded activation to behavior targets, supporting token-distribution targets.
- `MetricEstimator`: utilities for decoder Jacobian pullback metrics and behavior-head pullback metrics.
- `GeodesicSolver`: shortest-path approximation over latent coordinates under selectable metrics (`latent_linear`, `decoder_pullback`, `behavior_pullback`).
- `LossBundle`: reconstruction, KL, behavior alignment, isometry, geodesic naturalness, and optional patch/intervention consistency terms with all weights supplied by Hydra config.

## Inputs

- Activation/features tensor from upstream `subspace` or direct activation collector.
- Behavior targets from `baseline` per-class output distributions and/or patched-model outputs.
- Task causal embeddings/known topology metadata when available.

## Outputs

- Checkpoint with encoder, decoder(s), router, behavior head, and training config.
- Metrics JSON containing train/validation reconstruction, KL, behavior loss, isometry loss, chart usage entropy, and geodesic diagnostics.
- Latent coordinate tensors for train/test examples.
- Utility API with `encode`, `decode`, `project`, `metric`, and `geodesic` methods compatible with analysis-side steering code.

## Constraints

- No disk paths inside methods except explicit paths passed by the analysis layer.
- No hardcoded training defaults inside method code; defaults live in session-local Hydra analysis config.
- Patching the frozen model belongs in the analysis layer, not this method.
