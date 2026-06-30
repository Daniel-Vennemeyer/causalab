# Manifold-VAE comparison — Debug/baseline milestone report

**Session:** `2026-05-14--manifold-vae-comparison--lucid-geode`
**Status:** Debug pass complete and validated end-to-end. This report covers the **current-code spline baseline** + **one recon-only VAE arm**. The behavior-aligned / metric / atlas / known-topology sweep (the core scientific test) is not yet run.
**Run:** Llama-3.1-8B on `tars` (GPU 3); artifacts under `/data/jiang/vennemdp/causalab/<session>/artifacts/natural_domains_arithmetic_weekdays/llama31_8b/weekdays/`.

> Note: written from the run logs + result artifacts (metrics/CSV/JSON). Figures live on `/data` on tars and are referenced by path, not embedded. For a figure-embedded report, run `/interpret-experiment` on tars where the artifacts resolve.

## Objective (recap)

Does a behavior-aligned manifold VAE — explicit intrinsic coordinates, atlas charts, behavior-pullback geometry — yield a more faithful activation→behavior manifold and more natural steering paths than the current spline pipeline? (See `plan/RESEARCH_OBJECTIVE.md`.) This milestone establishes the **baseline endpoints** of that comparison on the structured `weekdays` domain.

## Setup

| | |
|---|---|
| Task | `natural_domains_arithmetic` / weekdays, `target_variable=result`, 7 classes (S¹), 49 enumerated inputs |
| Model | `meta-llama/Llama-3.1-8B`, layer 28, last-token residual stream |
| Subspace | PCA, k=64 |
| Output (belief) manifold | spline in Hellinger space, **intrinsic_dim=1 (S¹), 7 centroids** — correct topology recovered |
| VAE debug arm | flat_vae, latent_dim=2, unstructured topology, metric=latent_linear, recon+KL only (`w_behavior=w_isometry=w_geodesic=w_patch=0`), seed 42 |

## Pre-flight gate — PASS

Baseline accuracy **0.918** (45/49 correct), prob-accuracy 0.568. The model performs the weekday-arithmetic task well, so the behavior distributions reflect task behavior and the manifold comparison is meaningful. (`counterfactual_sanity.json` was empty — worth confirming on the next run, but accuracy alone clears the gate.)

## Current-code spline baseline — geometric (manifold geodesic) vs linear

Steering along the spline's manifold geodesic is **dramatically more behavior-natural than straight-line interpolation**, on every criterion, all highly significant (paired t-test across 50 pairs):

| Criterion | Geometric | Linear | Δ (geo−lin) | paired-t p |
|---|---|---|---|---|
| Coherence (mean P on-concept) ↑ | **0.779** ±0.003 | 0.717 ±0.005 | +0.062 | 6.7e-16 |
| Coherence (worst-step) ↑ | **0.740** ±0.005 | 0.667 ±0.006 | +0.073 | 2.3e-16 |
| Dist. from behavior manifold ↓ | **0.325** ±0.019 | 1.397 ±0.130 | −1.07 | 2.0e-10 |
| Dist. (geodesic ref) ↓ | **0.673** ±0.038 | 5.82 ±0.578 | −5.14 | 3.9e-12 |
| Isometry (Pearson r) ↑ | **0.990** | 0.887 | +0.103 | — |

Reading: the spline's manifold geodesics stay ~4× closer to the behavior manifold and are near-perfectly isometric to behavior geometry (r=0.99). This reproduces the paper's central claim for the spline and sets the **ceiling** the VAE must approach.

## VAE debug arm (flat, recon-only)

| Metric | Value | Notes |
|---|---|---|
| Reconstruction (feature MSE) | 72.16 | not directly comparable to the spline's KL-recon (different definition) |
| KL | 5.92 | |
| **Isometry (Pearson r)** | **0.066** | activation geometry essentially **unaligned** with behavior |
| Geodesic naturalness (off-manifold energy) ↓ | 9.68 | reference value for future arms |
| Behavior distance | null | no behavior head trained (recon-only arm) |
| Patch consistency | null | `patch_eval=false` (deferred) |

## Head-to-head

| Arm | Isometry r | Δ vs spline |
|---|---|---|
| spline (geometric) | **0.990** | — |
| flat VAE (recon-only, latent-linear, unstructured) | **0.066** | **−0.924** |

**Headline:** a reconstruction-only VAE recovers activation *density* (it reconstructs) but essentially **none of the behavior-aligned geometry** — isometry ≈ 0 vs the spline's ≈ 1.

## Hypothesis assessment

- **H1** (recon-only flat VAE produces less behaviorally natural interpolations than the spline): **Supported** at the geometry level — isometry 0.066 ≪ 0.990. Path-level behavior metrics for the VAE require a behavior-aligned arm (next).
- **H2** (behavior-aligned training improves steering): **Not yet tested** — needs the behavior-aligned arms.
- **H3** (decoder- vs behavior-pullback metric): **Not yet tested** — debug arm used `latent_linear` only.
- **H4** (single chart vs atlas): **Not yet tested**.

## Success criteria

- ✅ Current-code reproduction writes the full weekdays artifact set (baseline, subspace, activation_manifold, output_manifold, path_steering).
- ◑ VAE writes comparable artifacts — 1 debug arm done; the four priority ablations are pending.
- ◑ Same-axes comparison — isometry is computed on both sides and comparable; coherence/dist-from-behavior exist for the spline; behavior_distance & patch are null for the recon-only arm by design.
- ⏳ "VAE is an improvement only if behavior/path metrics improve without large recon regression" — not yet decidable (recon-only arm is not expected to improve; it's the floor).
- ✅ Negative-result value: the debug arm cleanly identifies that **reconstruction alone does not recover behavior geometry** — exactly the gap the sweep is designed to close.

## Caveats & fairness

- The spline manifold is **constructed on per-class centroids** (its control points *are* the class means), so its geodesics are behavior-ordered almost by definition → isometry ≈ 1 is partly structural. The recon-only VAE organizes its latent purely to reconstruct, with no behavior signal, so 0.066 is the **floor**, not a failure. The scientific question is how much the behavior-aligned losses / behavior-pullback metric / known-S¹ topology claw back.
- Reconstruction numbers are not cross-comparable across arms (VAE feature-MSE vs spline output-KL); isometry, coherence, geodesic-naturalness, and distance-from-behavior-manifold are the comparable axes.
- `ablation_matrix.md` currently blends spline+VAE in the "single-chart" / "recon-only" rows (both arms share those attribute values with only 2 arms present); this separates once the sweep adds metric/atlas/behavior-aligned arms.
- Patch/intervention consistency was not evaluated (`patch_eval=false`); it remains a validation arm for later.

## Artifacts (on tars, under `/data/.../weekdays/`)

- `baseline/accuracy.json`, `subspace/pca_k64/...`
- `output_manifold/spline_s0.0/result/manifold_spline/ckpt_final.safetensors`
- `path_steering/pca_k64/L28_last_token/spline_s0.0/result/criteria/{isometry,coherence,distance_from_behavior_manifold}/{geometric,linear}/metrics.json`
- `behavior_manifold_vae/pca_k64/L28_last_token/flat_vae_topo-unstructured_metric-latent_linear_charts1_seed42/result/{comparison_ready.json,metrics.json,latents.safetensors,ckpt_final.*}`
- `compare_manifold_architectures/default/{summary.csv,ablation_matrix.md,metric_deltas.json,figures/}`

## Next steps

Run the **priority sweep** to test H2–H4 against this baseline:
1. Behavior-aligned arm (`w_behavior>0`, `w_isometry>0`) — does behavior signal move isometry off the floor?
2. Behavior-pullback vs decoder-pullback metric (H3).
3. Atlas / multi-chart (H4).
4. Known **S¹** topology (the structured-domain sanity check — expected to help most on weekdays).

All at `batch_size=128`. The comparison machinery (training → metrics → aggregation) is validated and ready; the sweep arms drop into the same shared experiment root and re-aggregate via `compare_architectures`.
