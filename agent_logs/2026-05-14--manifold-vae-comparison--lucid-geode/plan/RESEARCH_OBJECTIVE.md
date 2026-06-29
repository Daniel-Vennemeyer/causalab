# Research Objective

## Session

**Session:** `agent_logs/2026-05-14--manifold-vae-comparison--lucid-geode/`

## Objective

Determine whether a behavior-aligned manifold VAE, with explicit intrinsic coordinates, atlas variants, and behavior-pullback geometry, yields more faithful activation-to-behavior manifolds and steering paths than the current causalab spline/flow manifold pipeline.

## Motivation

The current code can fit activation manifolds after subspace discovery and evaluate path steering, but its default spline formulation interpolates centroid geometry rather than training an end-to-end generative manifold that is behavior-aligned during fitting. The proposed VAE architecture tests the stronger claim that useful steering needs a learned activation manifold whose geometry is anchored in behavioral consequences, not merely smooth activation reconstruction. The ablations are designed to separate density reconstruction, intrinsic topology, chart structure, and behavior-pullback metrics so the conclusion is about manifold geometry rather than a generic latent autoencoder.

## Scope boundaries

- Out-of-scope behavior: open-ended sycophancy/refusal tasks are not the first-run target; they are a later stress test after the VAE works on structured toy domains.
- Out-of-scope model: models larger than the existing configured 8B local baseline are optional follow-ups unless the user explicitly allocates cluster time.
- Out-of-scope implementation: shipped `causalab/` code is not edited during the research planning phase; VAE prototypes should live under this session's `code/` tree until promoted manually.
- Out-of-scope claim: this session does not try to prove a universal manifold-learning theorem; it tests empirical steering and geometry metrics on causalab tasks.

## Success criteria *(recommended)*

- Current-code reproduction writes a complete baseline artifact set for at least `weekdays` and one 2D task (`graph_walk_grid_5x5` or equivalent), including `baseline`, `subspace`, `activation_manifold`, `output_manifold`, and `path_steering` outputs.
- The VAE analysis writes comparable artifacts for the four priority ablations: flat vs metric geometry, single chart vs atlas, reconstruction-only vs behavior-aligned training, and activation-metric vs behavior-pullback geodesics.
- For each task and ablation arm, the report can compare reconstruction error, behavior-alignment distance, isometry score, path coherence, distance from behavior manifold, and patch/intervention consistency on the same held-out pairs.
- The VAE is considered an improvement only if it improves behavior/path metrics without a large reconstruction or intervention-naturalness regression relative to the current spline baseline.
- A negative result is still successful if it cleanly identifies which added component fails to improve over the current pipeline and whether the failure is due to task simplicity, optimization, topology, or patching cost.

## Hypotheses *(recommended)*

- **H1.** Reconstruction-only flat VAE latents will match or approach spline reconstruction on simple tasks but produce less behaviorally natural interpolations. *Falsified if* flat reconstruction-only VAE paths match the behavior-aligned VAE on isometry, coherence, and patch consistency.
- **H2.** Behavior-aligned training will improve steering metrics over reconstruction-only training at similar reconstruction quality. *Falsified if* behavior-aligned losses do not improve held-out behavior distance or path metrics after matching model capacity and training budget.
- **H3.** Decoder-induced metrics will be sufficient for simple cyclic domains, but behavior-pullback metrics will be better on tasks where output distributions define the relevant geometry. *Falsified if* behavior-pullback geodesics do not improve behavior trajectory metrics on any selected task.
- **H4.** A single global chart will be enough for `weekdays` but an atlas will help on 2D graph-walk geometry or later heterogeneous behaviors. *Falsified if* the atlas adds no measurable benefit after controlling for parameter count and training time.
