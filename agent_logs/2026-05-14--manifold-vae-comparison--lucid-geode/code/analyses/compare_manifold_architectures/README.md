# Compare Manifold Architectures

Compare Manifold Architectures answers: *How do behavior-aligned VAE activation manifolds compare against the current spline manifold across reconstruction, behavior faithfulness, and geometry?* It aggregates every `behavior_manifold_vae` arm and the current-code spline baseline (`activation_manifold` + `path_steering`) into one comparison table, an ablation matrix, metric deltas, and figures — pure artifact aggregation, CPU-only, with no model or task model loaded. It sits at the end of the pipeline and consumes the artifacts the manifold analyses produce.

## Overview

```
behavior_manifold_vae/**/comparison_ready.json ──> VAE arm rows ──┐
activation_manifold/**/metadata.json (recon KL) ──┐                ├─> summary.csv
path_steering/{isometry,coherence,distance_*}/    ├─> spline row  ─┤   ablation_matrix.md
path_steering/results_summary.csv (optional)      ┘                │   metric_deltas.json (VAE − spline)
                                                                   └─> figures/{recon_vs_behavior,
                                                                                isometry_by_arm,
                                                                                geodesic_naturalness_by_arm}
```

Every input source is optional; missing metrics stay null and missing arms are reported rather than crashing.

## Configuration

**Root config** (`causalab/configs/config.yaml`) — shared params this analysis reads:
- `experiment_root` — output root and the scan root for all upstream arms.

**Module config** (`${SESSION_DIR}/code/configs/analysis/compare_manifold_architectures.yaml`):

```yaml
compare_manifold_architectures:
  _name_: compare_manifold_architectures
  _subdir: default
  _output_dir: ${experiment_root}/compare_manifold_architectures/${._subdir}
  metrics: [reconstruction, kl, behavior_distance, isometry_pearson_r, geodesic_naturalness, patch_consistency]  # the shared metric schema
  baseline_arm: spline        # which architecture is the delta baseline
  visualization:
    figure_format: pdf         # figure file format for the figures/ outputs
```

## Outputs

Rooted at `${experiment_root}/compare_manifold_architectures/default/`.

### Interpretation

- **`summary.csv`** — one row per arm (spline baseline + each VAE arm); columns are arm descriptors followed by the metric union (missing values blank). The at-a-glance comparison table.
- **`ablation_matrix.md`** — the six planned ablations (flat-vs-manifold-geometry, single-chart-vs-atlas, recon-only-vs-behavior-aligned, decoder-vs-behavior-pullback-metric, single-site-vs-trajectory, unstructured-vs-known-topology), each mapped to which arms are present and their mean reconstruction / isometry; cells read "not yet run" when no arm covers that axis.
- **`metric_deltas.json`** — per VAE arm, `vae_value − spline_baseline_value` for every metric present on both sides (pairs where either is null are skipped). Positive/negative direction depends on the metric.
- **`figures/`** — reconstruction-vs-behavior scatter (colored by architecture), isometry bar by arm, geodesic-naturalness bar by arm. Each plot is skipped (with a logged note) when it has fewer than one finite point.
- **`missing_inputs.json`** — written only when no arms are found at all; lists which upstream directories were absent.

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `summary.csv` | one row per arm; descriptor + metric columns | human reference / reporting |
| `ablation_matrix.md` | markdown table over the six ablations | human reference / reporting |
| `metric_deltas.json` | `{arm_id: {metric: delta}}` | human reference / reporting |
| `figures/*.{pdf,png}` | matplotlib scatter + bar charts | human reference |
| `missing_inputs.json` | presence flags + note (only when no arms) | provenance |
| `experiment_metadata.json` | resolved cfg snapshot | provenance |
