# Custom Analysis Spec: `behavior_manifold_vae`

## Purpose

Train and evaluate VAE-based activation manifolds as a drop-in comparison against current `activation_manifold`, `path_steering`, and `pullback` artifacts.

## Dependencies

- Required upstream: `baseline` and `subspace`.
- Optional upstream: `output_manifold` for behavior-space/geodesic comparison.
- Reads task causal embeddings and periodic/topology metadata when present.

## Config surface

- `method`: one of `flat_vae`, `metric_vae`, `atlas_vae`.
- `latent_dim`: intrinsic coordinate dimension.
- `topology`: `unstructured`, `s1`, `interval`, `r2`, `cylinder`.
- `n_charts`: number of atlas charts; ignored for non-atlas arms.
- `metric`: `latent_linear`, `decoder_pullback`, `behavior_pullback`.
- `loss_weights`: `recon`, `kl`, `behavior`, `isometry`, `geodesic`, `patch`.
- `activation_site`: `single_final_token` initially; later `trajectory`.
- `patch_eval`: boolean controlling expensive frozen-model patch validation.
- `selected_pairs` / `max_pairs`: match `path_steering` pair semantics.

## Outputs

Rooted at `${experiment_root}/behavior_manifold_vae/${_subdir}/`:

- `ckpt_final.pt` or safetensors equivalent.
- `metadata.json` with model size, latent dimension, chart count, topology, metric, and loss weights.
- `metrics.json` with reconstruction, KL, behavior distance, isometry, geodesic naturalness, and patch consistency.
- `latents.safetensors`.
- `paths/{metric}/pair_distributions.safetensors`.
- `visualization/manifold_3d.html` and per-pair path plots where possible.
- `comparison_ready.json` summarizing metrics with names aligned to current `path_steering` outputs.

## Pre-flight gates

- Upstream baseline accuracy must be high enough for behavior geometry to be meaningful.
- Upstream subspace must produce stable features at the selected layer/token site.
- A debug run must complete on a small pair subset before any full ablation sweep.

## Notes

This analysis should initially reuse the existing subspace feature surface so the comparison isolates manifold learning. Direct residual-stream decoding and multi-site trajectory activations are a later ablation after the single-site version is stable.
