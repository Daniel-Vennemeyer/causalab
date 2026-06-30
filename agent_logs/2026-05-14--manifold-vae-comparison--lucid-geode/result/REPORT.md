# Manifold-VAE comparison — Debug/baseline milestone report

**Session:** `2026-05-14--manifold-vae-comparison--lucid-geode`
**Status:** Debug pass + **priority sweep complete** (7 arms: spline ceiling, recon-only floor, and 5 behavior-aligned arms covering ablations 2/3/4a/4b/6). Single seed, weekdays only. Patch-grounded behavior metrics for the VAE are NOT yet computed (`patch_eval=false`).
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

**Headline (debug):** a reconstruction-only VAE recovers activation *density* but essentially **none of the behavior-aligned geometry** — isometry ≈ 0 vs the spline's ≈ 1.

## Priority sweep results (5 behavior-aligned arms)

All weekdays, L28, seed 42, shared root. `metric_vae` trains the *same* VAE as `flat_vae`; the `metric` knob only changes the geodesic used to score isometry — so the three rows with identical recon/kl/behavior (76.92 / 5.87 / 0.357) are a **controlled metric ablation on one fixed manifold**.

| arm | recon | kl | behavior_dist | isometry r | geo-nat |
|---|---|---|---|---|---|
| spline (ceiling) | — | — | — | **0.990** | — |
| flat, recon-only (floor) | 72.2 | 5.92 | — | 0.066 | 9.68 |
| flat, aligned, latent-linear | 76.9 | 5.87 | 0.357 | 0.204 | 9.25 |
| metric, aligned, decoder-pullback | 76.9 | 5.87 | 0.357 | **0.244** | 9.41 |
| metric, aligned, behavior-pullback | 76.9 | 5.87 | 0.357 | 0.171 | 8.87 |
| atlas (K=4), aligned, behavior-pullback | 64.8 | 5.81 | 0.335 | −0.027 | 8.05 |
| flat, aligned, **S¹** topology | **60.7** | 3.59 | 0.330 | −0.274 | 11.07 |

Findings (read against the caveats below):
- **Behavior alignment helps** the flat VAE: isometry 0.066 → 0.204.
- **Metric (Ablation 4):** decoder-pullback (0.244) > latent-linear (0.204) > behavior-pullback (0.171) — opposite of the H3 expectation that behavior-pullback is more faithful.
- **Atlas (Ablation 2):** no benefit on clean S¹ (≈0), as expected.
- **Known S¹ topology (Ablation 6):** best reconstruction (60.7, lowest KL 3.59) but **worst isometry (−0.27)** — the circle fits the cyclic density, but the 7 classes are ordered around it in a non-behavior-aligned way.
- **No VAE arm approaches the spline's 0.99** on the isometry proxy.

## Hypothesis assessment

- **H1** (recon-only flat VAE less behaviorally natural than spline): **Supported** at the geometry level — isometry 0.066 ≪ 0.990.
- **H2** (behavior-aligned training improves over recon-only): **Weakly supported** — 0.066 → 0.204 isometry, but the absolute level stays far below the spline and the effect size is within plausible single-seed noise.
- **H3** (behavior-pullback metric more faithful than decoder-pullback): **Not supported (provisionally refuted)** — decoder-pullback scored higher (0.244 vs 0.171). Differences are within noise; needs seeds.
- **H4** (atlas helps): **Not supported on weekdays** — atlas ≈ 0 isometry; expected (clean S¹ needs no atlas). Re-test on 2D graph_walk.
- **Bonus (topology):** imposing S¹ aids *density* (recon/KL) but **not** behavior-ordering — a clean, interpretable negative.

## Success criteria

- ✅ Current-code reproduction writes the full weekdays artifact set (baseline, subspace, activation_manifold, output_manifold, path_steering).
- ◑ VAE writes comparable artifacts — 1 debug arm done; the four priority ablations are pending.
- ◑ Same-axes comparison — isometry is computed on both sides and comparable; coherence/dist-from-behavior exist for the spline; behavior_distance & patch are null for the recon-only arm by design.
- ⏳ "VAE is an improvement only if behavior/path metrics improve without large recon regression" — not yet decidable (recon-only arm is not expected to improve; it's the floor).
- ✅ Negative-result value: the debug arm cleanly identifies that **reconstruction alone does not recover behavior geometry** — exactly the gap the sweep is designed to close.

## Caveats & fairness (these materially limit the conclusions)

1. **Single seed, 21 class-pairs.** Isometry is a Pearson r over W·(W−1)/2 = 21 pairs. Differences among the aligned arms (0.171 / 0.204 / 0.244) and the negatives (≈ 0 for atlas/S¹) are very likely **within noise**. No ranking of metrics/topologies is trustworthy until we have multiple seeds with error bars.
2. **The isometry comparison is stacked toward the spline.** Its control points *are* the class centroids, so it is structurally near-1; the VAE must *learn* that ordering from 49 examples. Isometry alone is therefore a weak basis for "spline beats VAE."
3. **The VAE is compared only on the geometry proxy.** The behavior-grounded path metrics (coherence, distance-from-behavior-manifold) that the spline scores well on come from *patching steered paths into the model*. With `patch_eval=false` we have none for the VAE — so the apples-to-apples behavioral comparison is **missing**, not lost.
4. **Tiny data.** 49 enumerated inputs / 7 classes; the behavior head (behavior_dist ≈ 0.33–0.36) is weak. Reconstruction is not cross-comparable across arms (VAE feature-MSE vs spline output-KL).

## Artifacts (on tars, under `/data/.../weekdays/`)

- `baseline/accuracy.json`, `subspace/pca_k64/...`
- `output_manifold/spline_s0.0/result/manifold_spline/ckpt_final.safetensors`
- `path_steering/pca_k64/L28_last_token/spline_s0.0/result/criteria/{isometry,coherence,distance_from_behavior_manifold}/{geometric,linear}/metrics.json`
- `behavior_manifold_vae/pca_k64/L28_last_token/flat_vae_topo-unstructured_metric-latent_linear_charts1_seed42/result/{comparison_ready.json,metrics.json,latents.safetensors,ckpt_final.*}`
- `compare_manifold_architectures/default/{summary.csv,ablation_matrix.md,metric_deltas.json,figures/}`

## Provisional conclusion

On weekdays, behavior-aligned training gives the VAE a **modest** isometry gain over recon-only (0.066 → ~0.20), but **no VAE arm recovers the behavior-aligned geometry the way the centroid-constructed spline does**, and the most "principled" knobs (behavior-pullback metric, imposed S¹) did **not** help the geometry proxy — S¹ even improved density while *hurting* behavior-ordering. Per the plan's success criteria, this is a useful (partly negative) result: it cleanly localizes where the added components fail to help on a clean structured domain. **But** conclusions are gated by single-seed noise, a spline-favoring proxy metric, and the absence of patch-grounded behavior metrics for the VAE (caveats above). This is suggestive, not yet decisive.

## Next steps (in priority order)

1. **Multiple seeds (≥3–5)** for every arm → error bars on isometry. Without this we cannot rank decoder vs behavior-pullback or call S¹ a regression.
2. **Enable `patch_eval`** → patch VAE-decoded steered paths into the model and score coherence / distance-from-behavior-manifold, the *same* axes the spline wins on. This is the fair behavioral comparison and the real test of the central claim.
3. **2D `graph_walk_grid_5x5`** → where atlas / multi-chart and richer topology are expected to matter (weekdays is too simple to need them).
4. Consider a stronger behavior head / more data, and an isometry definition that doesn't structurally favor the centroid spline (e.g., score both manifolds on learned, non-centroid coordinates).
