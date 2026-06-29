# Custom Analysis Spec: `compare_manifold_architectures`

## Purpose

Aggregate current-code and VAE manifold results into one comparison table and report-ready artifact set.

## Inputs

- Current spline/flow artifacts from `activation_manifold`, `path_steering`, and optional `pullback`.
- VAE artifacts from `behavior_manifold_vae`.
- Baseline/output-manifold artifacts for task accuracy and behavior-space reference distances.

## Outputs

Rooted at `${experiment_root}/compare_manifold_architectures/`:

- `summary.csv` with one row per task, architecture arm, metric, and seed.
- `ablation_matrix.md` summarizing the six ablations and their available metrics.
- `metric_deltas.json` comparing VAE arms against the current spline baseline.
- Figures under `figures/` for reconstruction vs behavior alignment, isometry vs patch consistency, and geodesic naturalness by arm.

## Comparison keys

- Task: `weekdays`, `months` optional, `graph_walk_grid_5x5` or equivalent 2D task.
- Architecture: current spline, current flow if run, flat VAE, metric VAE, atlas VAE.
- Training objective: reconstruction-only vs behavior-aligned.
- Metric: latent-linear, decoder-pullback, behavior-pullback.
- Activation surface: single layer/final token first; trajectory later.

## Failure handling

If no existing current-code artifacts are present, the analysis should report missing inputs explicitly and wait for `/run-experiment` to produce session-local current-code runs rather than falling back to global paths.
