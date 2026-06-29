# Objective Input

The user wants a full comparison between the current causalab behavior-manifold pipeline and a new Behavior-Aligned Manifold VAE architecture.

Raw objective, normalized to ASCII for this artifact:

- Learn an activation manifold M_h in R^{d_model} aligned with a behavior manifold M_y.
- The target is not a steering vector or subspace, but a parameterized manifold with usable geodesics.
- Proposed architecture:
  - collect frozen-model activations h(x), initially from one layer and final/answer token residual stream;
  - encode activations with q_phi(z | h), where z represents intrinsic manifold coordinates;
  - optionally route examples through an atlas of local charts r_phi(k | h), z_k, f_{theta,k}(z_k);
  - decode through an immersion map f_theta: Z -> M_h to produce h_hat;
  - patch h_hat into the frozen model to measure behavior y_hat;
  - use a learned behavior head for cheap training and patched-model behavior for causal validation;
  - estimate geometry either with decoder pullback metric G_h(z) = J_f(z)^T J_f(z) or behavior-pullback metric G_y(z) = J_{F o f}(z)^T g_y J_{F o f}(z).
- Proposed losses:
  - reconstruction;
  - KL;
  - behavior alignment;
  - isometry;
  - geodesic naturalness;
  - intervention consistency.
- Proposed ablations:
  1. flat latent VAE vs learned manifold geometry;
  2. single chart vs atlas/multi-chart;
  3. reconstruction-only VAE vs behavior-aligned/isometry/intervention-trained VAE;
  4. decoder-induced activation metric vs behavior-pullback metric;
  5. single activation site vs multi-layer/multi-token trajectory;
  6. unstructured latent topology vs known topology such as S1, interval, R2 sheet, or cylinder.
- The four highest-priority ablations are the first four above.
