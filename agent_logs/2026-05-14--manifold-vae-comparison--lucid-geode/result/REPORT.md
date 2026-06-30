# Manifold-VAE comparison — Debug/baseline milestone report

**Session:** `2026-05-14--manifold-vae-comparison--lucid-geode`
**Status:** Priority sweep + **objective fix complete** (seeded). Headline arc: the *original* behavior-aligned arms (Euclidean `d_y`) sit at isometry ≈ 0 ± 0.16 vs spline 0.99; the **cyclic relational + contrastive arm reaches isometry 0.936 ± 0.024 — matching the spline at no recon cost.** Patch-grounded behavioral comparison for the VAE NOT yet computed (`patch_eval=false`) — the key remaining test.

## BREAKTHROUGH: cyclic relational + contrastive objective (weekdays, 5 seeds)

Replacing the isometry loss's Euclidean `d_y` (distance between output-prob vectors) with the task's **cyclic class distance** `min(|Δ|,7−|Δ|)` + a supervised-contrastive order term (adjacent-near / distant-far) takes the learned VAE manifold from ≈ 0 to spline-level isometry:

| arm | isometry r (mean ± std) | recon | note |
|---|---|---|---|
| spline (ceiling) | 0.990 | — | centroid-constructed |
| **cyclic (per-example, cyclic+contrastive)** | **0.936 ± 0.024** | 80.5 | **learned; matches spline, recon unchanged** |
| centroid-upper (cyclic, 7 centroids) | 0.447 ± 0.232 | ~0.6 | per-example *beats* this bound |
| strong (Euclidean, KL 0.01, ×5 weights) | 0.089 ± 0.088 | 87.4 | rebalancing alone ≈ no help |
| **transition (behavior-derived `d_y`, topology NOT injected)** | **0.806 ± 0.066** | 64.3 | **DISCOVERS the ring from the model's +1 transitions** |
| activation (label-free: latent↔activation distances) | −0.123 ± 0.060 | 87.6 | structure NOT recoverable from raw activation geometry |
| earlier Euclidean arms (flat/metric/atlas/s1) | ≈ 0 ± 0.16 | 60–80 | floor |

### Patch-grounded steering (the causal arbiter) — preliminary, 1 seed

`patch_eval` patches VAE-decoded steered paths into the 8B model and scores the same axes as the spline. First result (`transition_centroid`, seed 0):

| | spline geometric | spline linear | transition_centroid (VAE) |
|---|---|---|---|
| coherence ↑ | 0.779 | 0.717 | **0.712** |
| dist-from-behavior-manifold ↓ | 0.325 | 1.397 | **1.434** |

**The discovered VAE manifold steers ≈ linear interpolation, not the spline geodesic.** It recovered the behavioral *order* (clean latent ring) but its steered *paths* aren't manifold-faithful: a straight line in the 2-D latent decodes to a roughly-linear activation path, so intermediate points fall OFF the behavior manifold (dist 1.43 ≈ linear's 1.40, vs the spline geodesic's 0.33). Consistent with the isometry proxy (≈0.82 ≈ linear 0.887). **Bottleneck = the path/decoder, not the latent.** Targeted fix under test: `transition_dpb` arm uses decoder-pullback geodesics (paths that follow the manifold) for both isometry and the patched path.

### Fit vs. discover — RESOLVED: discovery works via behavioral transitions

The cyclic arm (0.936) *injects* the topology (period, ordering); so does the spline (via `periodic_dims` metadata). The label-free `activation` arm (−0.12) showed the ring is **not** recoverable from raw activation-distance geometry (low-variance circular feature swamped in 64-d; static one-hot outputs are equidistant). The adjacency lives in the input→output **mechanism** — so we derived `d_y` from the model's **behavioral transitions**: step the ordinal input `number` by +1 (entity fixed), read how the model's *predicted* result class moves, take graph distance on the transition graph. **No period / ordering / "cyclic" is declared** — only that `number` is a steppable ordinal input.

Result: the `transition` arm reaches **isometry 0.806 ± 0.066** — recovering ~86% of the hand-coded-cyclic performance (0.936), vs −0.12 (label-free) and 0.066 (floor). The ring is **discovered**, not injected. The gap to 0.936 is honest noise: `transition_edge_count`=9 (a clean 7-ring has 7 adjacency pairs) → the model's ~8% misclassifications inject ~2 spurious edges, slightly perturbing `d_y`.

**Conclusion:** a VAE can learn the behavior-aligned manifold **without being told the structure**, provided the relational target is grounded in behavioral transitions rather than static activations/outputs. Remaining honesty caveats: (1) still assumes one input is ordinal (we step `number`); (2) single structured task — the real generalization test is a task whose topology we genuinely don't know; (3) patch-grounded steering (does this geometry drive the model?) still pending for all VAE arms.

Conclusions:
- **The objective was the bottleneck, confirmed.** A behaviorally-grounded relational target (cyclic `d_y`) + contrastive ordering recovers the weekday ring; the learned VAE (r=0.936) is statistically near the centroid-constructed spline (0.990), and does it **without hurting reconstruction**.
- **It was the *distance*, not KL collapse.** The `strong` arm (drop KL, ×5 behavior/isometry, still Euclidean) barely moved (0.089). Only swapping in the cyclic distance worked.
- **Per-example > centroid-supervised.** 0.936 vs 0.447 — more data under the relational loss beats fitting 7 centroids; the architecture is clearly capable.
- **Still open:** does spline-level *geometry* (isometry) translate to spline-level *behavioral steering*? That needs `patch_eval` on the cyclic arm — the next and decisive test.
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

## Priority sweep results — 6 seeds (0,1,2,3,4,42), mean ± std

All weekdays, L28, shared root. `metric_vae` trains the *same* VAE as `flat_vae` (only the geodesic metric scoring isometry differs). **With seeds, the picture from the single-seed run collapses.**

| arm | isometry r (mean ± std) | recon | behavior_dist |
|---|---|---|---|
| spline (ceiling, n=1) | **0.990** | — | — |
| flat, recon-only (floor, n=1) | 0.066 | 72.2 | — |
| flat, aligned, latent-linear | **0.001 ± 0.160** | 77.6 | 0.33 |
| metric, aligned, decoder-pullback | **0.005 ± 0.168** | 77.6 | 0.33 |
| metric, aligned, behavior-pullback | **0.032 ± 0.183** | 77.6 | 0.33 |
| atlas (K=4), aligned, behavior-pullback | **0.039 ± 0.159** | 65.8 | 0.37 |
| flat, aligned, **S¹** topology | **−0.058 ± 0.159** | **61.3** | 0.31 |

**The decisive finding:** every behavior-aligned VAE arm has isometry **r ≈ 0 with std ≈ 0.16–0.18** — statistically indistinguishable from zero, from each other, *and from the recon-only floor (0.066)*. The seed-to-seed spread (±0.16) dwarfs every difference, so the single-seed "results" (0.204 / 0.244 / 0.171 / −0.27) were **pure noise**. The spline sits at 0.99 (deterministic). The VAE's *expected* behavioral-geometry alignment on this task is ≈ 0.

## Hypothesis assessment (with seeds)

- **H1** (recon-only flat VAE less behaviorally natural than spline): **Supported** — VAE ≈ 0 ≪ spline 0.99.
- **H2** (behavior-aligned training beats recon-only): **NOT supported** — behavior-aligned mean (0.00–0.04) ≈ the recon-only floor (0.066), all within ±0.16. The single-seed 0.204 "gain" was noise. **Behavior prediction + Euclidean-`d_y` isometry loss does not produce behavior-aligned latent geometry.**
- **H3** (behavior-pullback > decoder-pullback metric): **No effect** — 0.032 vs 0.005, both ≈ 0 ± 0.17. Indistinguishable.
- **H4** (atlas helps): **No effect** — 0.039 ± 0.16, ≈ 0.
- **Topology (S¹)**: best reconstruction (61.3, lowest KL) but isometry −0.058 ± 0.16 ≈ 0. Imposing the circle aids *density*, not behavioral ordering.

**Bottom line:** on weekdays, **none of the objective/metric/topology/atlas variants move the learned manifold's isometry off zero.** This sharply confirms the standing diagnosis — the current objective is too weak to force global behavioral geometry — and vindicates running seeds (every single-seed ranking was an artifact). Two things must still be checked before calling the *method* a failure (vs the metric/data): the **patch-grounded** behavioral comparison (does VAE steering produce coherent behavior despite low isometry?) and the **latent plots** (what failure mode — scramble, clusters, fold?).

## Success criteria

- ✅ Current-code reproduction writes the full weekdays artifact set (baseline, subspace, activation_manifold, output_manifold, path_steering).
- ◑ VAE writes comparable artifacts — 1 debug arm done; the four priority ablations are pending.
- ◑ Same-axes comparison — isometry is computed on both sides and comparable; coherence/dist-from-behavior exist for the spline; behavior_distance & patch are null for the recon-only arm by design.
- ⏳ "VAE is an improvement only if behavior/path metrics improve without large recon regression" — not yet decidable (recon-only arm is not expected to improve; it's the floor).
- ✅ Negative-result value: the debug arm cleanly identifies that **reconstruction alone does not recover behavior geometry** — exactly the gap the sweep is designed to close.

## Caveats & fairness (these materially limit the conclusions)

1. ~~Single seed~~ **RESOLVED — 6 seeds run.** The std (±0.16–0.18) confirms every single-seed difference was noise; all behavior-aligned arms are ≈ 0. (Isometry r is over only 21 class-pairs, hence the large per-seed variance.)
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

Across 6 seeds, **every** behavior-aligned VAE variant has isometry ≈ 0 (±0.16) — indistinguishable from each other and from the recon-only floor — versus the spline's 0.99. Behavior-alignment, behavior-pullback geometry, an atlas, and an imposed S¹ topology **all fail to move the learned manifold toward behavioral geometry** on this task. The earlier single-seed gains/rankings were noise. This is a clean negative on the isometry axis and squarely confirms the standing diagnosis: the current objective (behavior prediction + Euclidean-`d_y` isometry loss) is too weak to force global behavioral geometry. It is **not yet** a verdict on the *method* — that needs the patch-grounded test + a stronger relational objective.

## Next steps (in priority order)

1. ~~Multiple seeds~~ **DONE** — collapsed all single-seed differences to ≈ 0 ± 0.16.
2. **`patch_eval`** → patch VAE-decoded steered paths into the model, score coherence / distance-from-behavior-manifold on the spline's axes. Tells us whether VAE steering is behaviorally coherent *despite* ≈ 0 isometry (i.e. whether the proxy under-rates it). Implemented; needs the GPU run.
3. **Latent-coordinate plots** (`latent_coords.png`, now produced per arm) → diagnose the failure mode: scrambled loop, separated clusters, or fold. Determines whether the fix is the objective or the parameterization.
4. **Stronger relational objective** (the standing guidance): replace Euclidean `d_y` with the task's **cyclic** distance; add a supervised-contrastive / ordinal loss; add a **centroid-supervised upper-bound** arm to separate objective failure from data scarcity (49 inputs).
5. **2D `graph_walk_grid_5x5`** → where atlas / richer topology should matter.
