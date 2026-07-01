# Manifold-VAE comparison — Debug/baseline milestone report

**Session:** `2026-05-14--manifold-vae-comparison--lucid-geode`
**Status:** Priority sweep + objective fix + steering fix + **cross-domain generalization** (seeded). Headline generalization result: **discovery generalizes to every topology** (isometry 0.73–0.84 across weekdays/months cyclic AND alphabet/age linear), and the **spline's geodesic advantage is universal** (beats linear ~2–4× on every topology, matching Wurgaft et al. — weekdays/months/alphabet). The VAE captures that advantage partially; `w_manifold` is the one component whose sign is topology-dependent (helps cycles, hurts lines), handled by an adaptive gate. Two coordinate/periodicity bugs in the *reference* manifolds were found and fixed (parameter-mode belief AND activation fits for non-cyclic domains); the alphabet spline went from isometry 0.058 → 0.9995 after the fix, retracting an earlier "flat manifolds have no geodesic advantage" claim. Headline arc: the *original* behavior-aligned arms (Euclidean `d_y`) sit at isometry ≈ 0 ± 0.16 vs spline 0.99; the **cyclic relational + contrastive arm reaches isometry 0.936 ± 0.024 — matching the spline at no recon cost**; and the **manifold-interpolation loss (`w_manifold`) is the first VAE arm to move steering — distance 1.32 → 0.99 with its own decoded paths and coherence tied with the spline (0.770 vs 0.779)**, though it does not close the full gap to the spline's 0.325, and curving the path along the decoder-pullback geodesic makes it *worse* (1.54), not better.

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

### Patch-grounded steering (the causal arbiter) — 3 seeds, DEFINITIVE

`patch_eval` patches VAE-decoded steered paths into the 8B model and scores the same axes as the spline.

| arm | isometry | coherence ↑ | dist-from-behavior-manifold ↓ |
|---|---|---|---|
| **spline geometric** | 0.990 | **0.779** | **0.325** |
| spline linear | 0.887 | 0.717 | 1.397 |
| cyclic_centroid (injected, iso 0.91) | 0.909 | 0.705 ± 0.013 | 1.280 ± 0.067 |
| transition_centroid (discovered, latent-linear path) | 0.820 | 0.717 ± 0.005 | 1.317 ± 0.136 |
| transition_dpb (discovered, on-manifold geodesic) | 0.823 | 0.715 ± 0.004 | 1.426 ± 0.115 |

**Every VAE arm steers like linear interpolation (coherence ≈ 0.71, dist ≈ 1.3–1.4); none approaches the spline geodesic (0.78 / 0.33).** This is robust to isometry (0.82→0.91), path metric (latent-linear vs decoder-pullback geodesic — the on-manifold geodesic did NOT help, dist 1.43), and injected-vs-discovered structure.

**Confirmation test (`transition_project`, 3 seeds):** snapping each decoded path point to the nearest REAL training activation (same discovered ordering, patched via the subspace featurizer) drops distance-from-behavior-manifold **1.32 → 0.772 ± 0.030** and lifts coherence to **0.737 ± 0.011** — halfway from linear (1.40) to the spline geodesic (0.325), clearly beating linear. This **confirms the steering deficit is off-distribution decoder interpolants, not the discovered ordering**: with realistic path points, behavior stays near the manifold. It doesn't fully reach 0.325 because the snap is coarse (50 steps → ~49 real points, a step-function vs the spline's smooth interpolation). A decoder producing smooth in-distribution interpolants (bijective `flow`, or learned projection) would close the rest — the clear, well-motivated next investment.

Decisive insight: **isometry does not predict steering.** cyclic_centroid (iso 0.91) still steers at dist 1.28 ≈ linear. The spline wins because its geodesic interpolates **real class centroids in activation space** → realistic intermediate activations → on-manifold behavior. The VAE's MLP decoder is trained to reconstruct *data points* only; its *interpolated* path points are **off-distribution**, so patched intermediates drift off the behavior manifold like a straight line. "Shortest path in decoder-output space" (decoder-pullback) ≠ "stays on the realistic-activation manifold," which is why that fix failed. Closing this needs a decoder whose interpolants stay on the data manifold (e.g. the bijective `flow` method, or projecting decoded path points onto real activations) — a deeper architectural change, documented as future work.

## STEERING FIX: manifold-interpolation loss (`w_manifold`) — 3 seeds

Acting on the project-to-data confirmation (the deficit is off-distribution decoder interpolants, not the ordering), we added a **manifold-interpolation loss** `w_manifold`: during training, decode random latent interpolants `u_mid = u_a + t·(u_b − u_a)` and penalize each decoded point's distance to the nearest real activation (`min cdist`). This trains the decoder to keep *interpolated* outputs on the data manifold — an in-loss substitute for a manifold-faithful decoder, no architecture change. Arm `transition_manifold` = the discovered-transition recipe + `w_centroid_iso=5` + `w_compactness=1` + **`w_manifold=1`**, `patch_eval=true`.

| arm | decoder | path | isometry | coherence ↑ | dist ↓ | geo_nat ↓ | recon |
|---|---|---|---|---|---|---|---|
| **spline geometric** | (real centroids) | geodesic | 0.990 | **0.779** | **0.325** | — | — |
| transition_centroid (no w_manifold) | recon-faithful | latent-linear | 0.818 | 0.717 | ~1.26 | 10.24 | 58.3 |
| transition_project *(crutch: snap to real)* | — | latent-linear | 0.824 | 0.737 | 0.772 | 10.25 | 58.3 |
| **transition_manifold** (`w_manifold=1`) | **on-manifold** | **latent-linear** | 0.876 ± .066 | **0.770 ± .030** | **0.988 ± .114** | **5.39** | 107.6 |
| transition_manifold_geo (`w_manifold=1`) | on-manifold | **pullback geodesic** | 0.766 | 0.742 | 1.538 ± .279 | 8.18 | 107.6 |

Per-seed dist for `transition_manifold`: 1.12 / 0.89 / 0.96 (tight, no lucky seed).

**Three findings:**

1. **`w_manifold` is the first reproducible VAE steering win.** Distance dropped **1.26 → 0.99** using the VAE's *own* decoded paths (no project-to-data crutch), and coherence rose to **0.770 — statistically tied with the spline (0.779)**. On the text-coherence axis the manifold-faithful VAE now steers as cleanly as the spline. `geodesic_naturalness` halved (10.2 → 5.4): decoded paths are far more on-manifold.

2. **Reconstruction MSE is the wrong objective for steering.** Recon *doubled* (58 → 108) while coherence went *up*. The decoder trades pointwise MSE for on-manifold interpolants — exactly what steering needs. This vindicates the "manifold-faithful, not reconstruction-faithful" framing directly.

3. **Path *routing* via the decoder-pullback geodesic is a dead end (refuted).** `transition_manifold_geo` is the same trained decoder (recon bit-identical per seed) with the patch path solved as a pullback geodesic instead of a straight line — and it is **worse** (0.99 → 1.54), with `geodesic_naturalness` rising (5.4 → 8.2). The decoder-pullback metric `G_h = JᵀJ` measures where the *decoder stretches*, not where *data lives*; its geodesic shortcuts through off-data low-stretch regions. With an on-manifold decoder, **the straight latent line is already near-optimal**; curving it via decoder geometry hurts.

**Remaining gap (0.99 → 0.325) is honest and localized.** Even the project-to-data crutch floors at 0.77, so the spline's edge is its path threading **dense real-activation regions in behavior order**, which neither a straight latent line nor a decoder-pullback geodesic reproduces. The next lever is a **data-density metric** (geodesics cheap where data is dense, expensive in voids — Arvanitidis-style RBF/uncertainty metric), *not* a decoder-Jacobian metric — documented as the next investment.

## GENERALIZATION across domains + topologies (Phase 1: months, alphabet, age)

Replicated the weekdays pipeline on three more `natural_domains_arithmetic` domains — a larger cycle (months, W=12) and two **non-cyclic lines** (alphabet W=22, age W=91) — to test whether the discovery + steering results are a weekday-ring artifact. Minimal decisive arm set (spline baseline + `transition_centroid` + `transition_manifold`), 3 seeds; age is discovery-only (W=91 makes patch infeasible). All metrics use `intrinsic_mode: parameter` belief manifolds (see the belief-fit bug below).

### Discovery generalizes to EVERY topology (the strong positive)

`transition_centroid` isometry (discovered `d_y` from behavioral `number` transitions; NO topology injected):

| domain | topology | W | isometry r (3 seeds) |
|---|---|---|---|
| weekdays | cyclic | 7 | 0.82 |
| **months** | cyclic | 12 | **0.84** |
| **alphabet** | **line (non-cyclic)** | 22 | **0.73** |
| **age** | **line (non-cyclic)** | 91 | **0.83** |

The VAE recovers the correct behavioral geometry — a ring for cyclic domains, a **line** for non-cyclic ones — from transitions alone, across W=7…91. Discovery is **not** topology-specific.

### The geodesic advantage is UNIVERSAL; only `w_manifold`'s benefit is topology-gated

> **CORRECTION.** An earlier version of this section claimed "flat manifolds have no geodesic advantage" and that "the VAE beats the spline on the linear topology." **Both were artifacts of a mis-fit alphabet spline** and are now retracted. The spline's *activation* manifold was fit on the top-1 PCA axis (`No periodic pairs — using top 1 PCA components`), which scrambles letter order when the curve folds → isometry 0.058 and an impossible geodesic>linear inversion. Fitting the activation manifold on the **ordinal-index coordinate** (`intrinsic_mode: parameter`, matching the paper's sequential-task protocol, App. A.3) fixes it: alphabet spline **isometry 0.058 → 0.9995** (paper letters: 0.999) and **geometric distance 1.22 → 0.219 ≪ linear 0.94**. The geodesic advantage exists on the line after all.

| domain | spline geometric ↓ | spline linear | VAE transition_centroid | VAE transition_manifold (`w_manifold`) |
|---|---|---|---|---|
| weekdays (cyclic) | 0.33 | 1.40 | 1.26 | **0.99** (helps) |
| **months** (cyclic) | 0.19 | 0.78 | 0.69 | **0.43** (helps) |
| **alphabet** (line) | **0.219** | 0.94 | 0.71 | 1.24 (hurts) |

- **The spline geodesic beats linear on EVERY topology** (weekdays 0.33 vs 1.40; months 0.19 vs 0.78; alphabet 0.219 vs 0.94), reproducing the paper's finding across cyclic *and* sequential concepts. There is no "flat = no advantage" exception — that was the coordinate bug.
- **The VAE captures the advantage partially but does not match the well-fit spline.** On alphabet the VAE `transition_centroid` (0.71) beats linear (0.94) but trails the spline geodesic (0.219); on months the VAE `transition_manifold` (0.43) beats linear (0.78) but trails the spline (0.19). The gap: the spline follows a smooth 1-D ordered curve exactly, while the VAE traces a straight line in a 2-D *unstructured* latent that only approximately respects order.
- **`w_manifold`'s benefit — not the geodesic advantage — is what's topology-gated.** It helps the VAE on cycles (weekdays 1.26→0.99, months 0.69→0.43) and hurts on the line (alphabet 0.71→1.24). Mechanism (revised): on a ring the straight latent line chords across the loop, so pulling decoded interpolants back onto the manifold helps net; on a line the straight-in-order latent path already tracks the manifold, so `w_manifold`'s nearest-neighbor pull only adds off-order jaggedness. The **adaptive gate** (gate `w_manifold` off on a discovered line) is the correct handling either way.

**Unified finding (corrected):** discovery is universal (isometry 0.73–0.84, all topologies); the **spline's geodesic advantage is universal** (beats linear everywhere, ~2–4×, matching the paper); the VAE is a discovery-driven approximation that captures that advantage partially; and `w_manifold` is the one component whose sign is topology-dependent, correctly handled by the adaptive gate.

### TWO reference-manifold bugs on non-cyclic domains (both found + fixed)

The non-cyclic domains had two *separate* bugs in the reference manifolds (not the VAE), both from unsupervised coordinate/periodicity heuristics misfiring on a line:

1. **Belief manifold periodicity (`output_manifold`).** The PCA heuristic `detect_periodic_dims` false-positived and wrapped alphabet's line E..Z onto a circle (`dim0=[0,2π]`, A≈Z). This corrupts the **VAE's** eval, whose D_Y is the belief manifold. Fix: `output_manifold.intrinsic_mode: parameter` (periodicity from `causal_model.periods`, periodic iff `cyclic=True`). After this, **VAE `transition_centroid` isometry rose −0.04 → 0.73**.
2. **Activation manifold coordinate (`activation_manifold`).** Even after (1), the *spline* isometry stayed at **0.058**: the spline's activation manifold was fit on the **top-1 PCA axis** (`No periodic pairs — using top 1 PCA components`), which scrambles letter order when the curve folds. Fix: `activation_manifold.intrinsic_mode: parameter` (ordinal-index coordinate, the paper's sequential protocol). After this, **spline isometry 0.058 → 0.9995 and geometric distance 1.22 → 0.219 ≪ linear 0.94** — matching the paper's letters (0.999; 2.42 ≪ 6.95).

Both fixes apply only to the non-cyclic domains; **weekdays/months (cyclic) use the PCA `atan2` coordinate, which is already the paper's cyclic protocol and was correct** (months re-confirmed under parameter mode). All pre-fix alphabet/age numbers were invalid; every alphabet figure above is post-fix.

## Final summary: fit → discover → steer

| question | answer |
|---|---|
| Can a VAE **fit** a known behavior manifold? | **Yes** — isometry 0.94 (cyclic, injected), no recon cost; ring emerges in an unstructured latent. |
| **Discover** it without injecting structure? | **Yes** — isometry 0.82 from behavioral transitions (step `number`, read class moves); ring recovered, no topology declared. |
| Beat **linear** on the isometry proxy? | Marginally (injected centroid arm 0.91 > 0.887; discovered 0.82 < 0.887). |
| **Steer the model** like the spline (the metric that matters)? | **Partly** — `w_manifold` (on-manifold decoder) is the first arm to move it: coherence 0.770 **ties the spline (0.779)** and dist 1.26 → **0.99** (own decoder). Still 3× the spline's 0.325 on dist; the residual is path *routing* through dense data, which a decoder-pullback geodesic does **not** fix (it makes it worse, 1.54). |

**Bottom line:** a VAE *discovers* the behavior manifold's structure from behavior alone (a real positive), and the steering gap to the spline is now **localized and partially closed**. The `w_manifold` loss confirms the diagnosis — an on-manifold decoder lifts coherence to spline parity (0.770 vs 0.779) and cuts distance to 0.99 *with the VAE's own decoded paths*, at the cost of reconstruction MSE (which proves recon was the wrong target). The residual 0.99→0.33 is **not** decoder-interpolant drift (that's fixed); it is that the spline's path threads **dense real-activation regions in behavior order**. Note: our `transition_manifold_geo` arm (which worsened to 1.54) tested the **decoder-Jacobian pullback *metric* geodesic (G_F)**, *not* Wurgaft et al.'s pullback, which is an **L-BFGS optimization of an activation path to match a behavior geodesic** (§3.3; it *succeeds*, letters R²=0.78). So we have *not* refuted pullback in general — only the decoder-metric-geodesic variant. **Candidate next levers:** the **density metric G_E** (Béthune 2025 energy-based Riemannian metric — the formalized version of the data-density idea), or the paper's optimization-based pullback.

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
