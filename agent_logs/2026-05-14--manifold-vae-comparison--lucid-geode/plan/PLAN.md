# Experiment Plan

## Section B. Causal model and dataset

This plan uses existing task packages rather than creating a new task in the first pass.

**Primary task:** `natural_domains_arithmetic_weekdays` via package `causalab/tasks/natural_domains_arithmetic/`
- **Status:** exists.
- **Why:** cyclic 1D behavior with known topology (`S1`), small output space, and existing runner preset `causalab/configs/runners/weekdays/weekdays_8b_pipeline.yaml`.

**Secondary task:** `graph_walk_grid_5x5` via package `causalab/tasks/graph_walk/`
- **Status:** exists.
- **Why:** 2D sheet-like behavior with longer context and a harder manifold surface; useful for atlas and behavior-pullback tests after the 1D debug pass.

**Optional sanity tasks:** `natural_domains_arithmetic_months` and `graph_walk_cylinder_9x9`
- **Status:** exists.
- **Why:** months gives a second cyclic domain; cylinder tests known topology `S1 x R` once the VAE implementation is stable.

### Causal variables

| Task | Name | Type | Cardinality | Sketch of value space |
|---|---|---:|---:|---|
| weekdays | `entity` | categorical/cyclic | 7 | weekday names |
| weekdays | `number` | ordinal/cyclic | 7 | `one` through `seven` |
| weekdays | `result` | categorical/cyclic | 7 | `(entity + number) mod 7` |
| graph walk grid | `node_coordinates` | numeric tuple | 25 | `(row, col)` on a 5x5 grid |

### Mechanism summary

- Weekdays: `result = (entity_index + number_value) mod 7`; `raw_input` asks what day is N days after the entity; `raw_output` is the result token.
- Graph walk: `node_coordinates` determines a random walk ending at that node; `raw_input` is the separator-joined walk; `raw_output` is the valid next-token neighbor set.

### Expected behavior

```text
Input:  "Q: What day is three days after Monday?\nA:"
Output: " Thursday"

Input:  "Q: What day is seven days after Sunday?\nA:"
Output: " Sunday"
```

For graph walk, examples are generated from the configured concept graph; the model must put probability mass on valid neighbor concepts after the final separator.

### Edge / stress cases

- Week wraparound, e.g. `Sunday + one -> Monday`, tests cyclic topology.
- `number = seven` for weekdays tests identity on `S1`.
- Grid corners and boundaries test asymmetric neighbor sets.
- Long graph-walk contexts test whether the activation site captures the terminal node rather than local token statistics.

### Counterfactual generator

| Task | Variable | `task.resample_variable` | Why |
|---|---|---|---|
| weekdays | `result` target with all inputs resampled | `all` | current runner uses centroid-style behavior comparisons and does not require pairwise single-variable CFs |
| graph walk | `node_coordinates` | `all` | one causal variable drives the task; centroid/path evaluations remain valid |

### Dataset sizing

```yaml
task:
  n_train: 1000
  n_test: 50
  enumerate_all: true   # weekdays/months
  balanced: false
```

For `graph_walk_grid_5x5`, use the existing task default `enumerate_all: false`, `n_train: 1000`, `n_test: 50`, and batch sizes of 8 because of 2048-token contexts.

## Section C. Neural surface

### Model(s)

| Model | Config | Why |
|---|---|---|
| `meta-llama/Meta-Llama-3.1-8B` | `model: llama31_8b` | existing runner presets target this model; feasible baseline for a full ablation before larger-model cluster runs |

### Activation surface

- First pass: single layer, final/answer-token residual stream, matching the current runner presets that pin layer 28.
- Current pipeline surface: `subspace -> activation_manifold`, usually with PCA/DAS features before spline fitting.
- VAE first-pass surface: reuse the same upstream subspace features so the comparison isolates manifold fitting and path geometry.
- Later ablation: direct residual-stream activations and multi-layer/multi-token trajectories once the single-site VAE is stable.

### Tokenization-check predictions

- Weekday and month result strings should be checked as one-token or scoreable output strings under Llama-3.1-8B before interventions.
- Graph-walk concepts are selected to be tokenizer-friendly, but `/run-experiment` should still validate concept token IDs before scoring.

### Compute budget

| Phase | Where | Expected wall time | GPUs |
|---|---|---:|---:|
| current weekdays reproduction | inline or slurm | 1-3 hours | 1 |
| VAE weekdays debug | inline or slurm | 30-90 minutes per arm | 1 |
| VAE weekdays priority ablations | slurm preferred | 6-18 hours total depending on pair count | 1 |
| graph-walk current reproduction | slurm preferred | 3-8 hours | 1 |
| graph-walk VAE ablations | slurm preferred | 12-36 hours depending on atlas/patch settings | 1 |

### Hardware constraints

- The plan assumes one 8B model run at a time.
- Patch-based VAE losses are expensive; use a behavior head during training and reserve frozen-model patching for validation/fine-tuning arms.
- No existing global `artifacts/` tree was found in this checkout, so current-code results should be rerun under this session unless the user supplies an older results path.

## Section D. Analysis-chain DAG

### DAG diagram

```text
baseline
  |\
  | \---------------------> output_manifold
  v                         |
subspace                    |
  |\                        |
  | \-> activation_manifold -> path_steering
  |                            |
  |                            v
  |                         pullback (optional)
  |
  v
custom: behavior_manifold_vae
  |
  v
custom: compare_manifold_architectures
```

### Node 1: `baseline`

- **Research question (scoped):** Is the model accurate enough on each task for behavior-manifold comparisons to be meaningful?
- **Method:** existing `baseline`.
- **Upstream artifacts consumed:** none.
- **Downstream artifacts produced:** `${experiment_root}/baseline/accuracy.json`, `per_class_output_dists.safetensors`, `counterfactual_sanity.json`.
- **Non-default knobs:** graph-walk batch size 8; otherwise use task defaults.
- **Pre-flight check:** accuracy and counterfactual sanity must be high enough that output distributions reflect task behavior rather than random failure. If weekdays accuracy is poor, stop.
- **Estimated runtime + GPU footprint:** minutes to 1 hour on 1 GPU for weekdays; longer for graph walk.

### Node 2: `subspace`

- **Research question (scoped):** What low-dimensional activation surface should both current and VAE manifold learners consume?
- **Method:** current subspace pipeline, initially using the existing runner defaults and layer 28.
- **Upstream artifacts consumed:** baseline outputs and collected activations.
- **Downstream artifacts produced:** `${experiment_root}/subspace/.../rotation.pt`, training features, metadata, and feature visualizations.
- **Non-default knobs:** graph walk uses `k_features: 64`, `batch_size: 8`, `token_positions: [last]`; weekdays uses layer 28.
- **Pre-flight check:** subspace artifacts must contain stable train features and metadata for the target layer/token.
- **Estimated runtime + GPU footprint:** 30 minutes to several hours on 1 GPU.

### Node 3: `activation_manifold` (current code baseline)

- **Research question (scoped):** How well does the current spline manifold fit activation geometry before adding a VAE?
- **Method:** current `spline` method, with optional `flow` arm if time allows.
- **Upstream artifacts consumed:** `subspace: null` to auto-discover the upstream subspace.
- **Downstream artifacts produced:** `${experiment_root}/activation_manifold/spline_s0.0/manifold_spline/ckpt_final.safetensors`, metadata, metrics, `visualization/manifold_3d.html`.
- **Non-default knobs:** `intrinsic_mode: parameter` and `intrinsic_dim: 2` for graph walk; `layers: [28]`; task-specific token position.
- **Pre-flight check:** reconstruction/residual metrics are finite and visualization/metadata writes succeed.
- **Estimated runtime + GPU footprint:** usually lighter than patch-based steering; minutes to an hour after subspace exists.

### Node 4: `output_manifold`

- **Research question (scoped):** What behavior-space geometry should activation geodesics align with?
- **Method:** current output-manifold spline over output distributions.
- **Upstream artifacts consumed:** baseline per-class output distributions.
- **Downstream artifacts produced:** `${experiment_root}/output_manifold/spline_s0.0/...`.
- **Non-default knobs:** graph walk uses `intrinsic_mode: parameter`, `intrinsic_dim: 2`, `batch_size: 8`.
- **Pre-flight check:** output manifold checkpoint and metadata exist for the target variable.
- **Estimated runtime + GPU footprint:** minutes to 1 hour.

### Node 5: `path_steering`

- **Research question (scoped):** How does the current spline manifold steer behavior along geometric and linear paths?
- **Method:** existing path steering with `isometry`, `coherence`, and `distance_from_behavior_manifold`.
- **Upstream artifacts consumed:** auto-discovered subspace and activation manifold; output manifold for dual/behavior-space plots.
- **Downstream artifacts produced:** `${experiment_root}/path_steering/.../results_summary.csv`, criteria metrics, per-pair distributions, and path visualizations.
- **Non-default knobs:** weekdays uses `n_extra_pairs: 29` and `isometry.n_interior_per_pair: 4`; graph walk uses smaller `n_prompts` and `num_steps_along_path` initially.
- **Pre-flight check:** a debug subset of pairs must complete before the full pair sample.
- **Estimated runtime + GPU footprint:** 1-8 hours depending on pair count and context length.

### Node 6: `pullback` (optional current-code reference)

- **Research question (scoped):** Can current code optimize activation trajectories that recapitulate output-manifold paths?
- **Method:** existing pullback analysis.
- **Upstream artifacts consumed:** activation manifold and output manifold.
- **Downstream artifacts produced:** `${experiment_root}/pullback/.../belief_paths/`, `embedding_paths/`, `activation_dist_from_manifold.json`, optimization caches.
- **Non-default knobs:** run only on weekdays or small selected graph-walk pairs initially.
- **Pre-flight check:** output manifold checkpoint auto-discovery must succeed.
- **Estimated runtime + GPU footprint:** potentially expensive; optional after path-steering results.

### Node 7: custom method `behavior_aligned_vae`

- **Research question (scoped):** Can a VAE expose usable intrinsic manifold coordinates and metrics while preserving activation reconstruction?
- **Spec path:** `${SESSION_DIR}/plan/setup_method_behavior_aligned_vae.md`
- **Upstream artifacts consumed:** none directly; consumed by the custom analysis.
- **Downstream artifacts produced:** method library under `${SESSION_DIR}/code/methods/behavior_aligned_vae/`.
- **Non-default knobs:** all training and loss defaults must live in analysis config, not method code.
- **Pre-flight check:** unit/debug training on cached subspace features reconstructs held-out features and writes a loadable checkpoint.
- **Estimated runtime + GPU footprint:** implementation first; debug training under 1 hour target.

### Node 8: custom analysis `behavior_manifold_vae`

- **Research question (scoped):** Which VAE architecture/loss/metric arm best matches behavior-aligned activation geometry?
- **Spec path:** `${SESSION_DIR}/plan/setup_analysis_behavior_manifold_vae.md`
- **Upstream artifacts consumed:** baseline, subspace, optional output manifold.
- **Downstream artifacts produced:** `${experiment_root}/behavior_manifold_vae/${_subdir}/ckpt_final.*`, `metrics.json`, `latents.safetensors`, path distributions, visualizations, `comparison_ready.json`.
- **Non-default knobs:** expose architecture arm, topology, chart count, loss weights, metric type, activation site, patch-eval controls, and selected pairs.
- **Pre-flight check:** debug run with one task, one arm, and a small pair subset writes comparison-ready metrics before the sweep.
- **Estimated runtime + GPU footprint:** 30-90 minutes per debug arm; full sweep depends on patch evaluation.

### Node 9: custom analysis `compare_manifold_architectures`

- **Research question (scoped):** Across matched tasks and pairs, does the Behavior-Aligned Manifold VAE beat or clarify the limits of the current pipeline?
- **Spec path:** `${SESSION_DIR}/plan/setup_analysis_compare_manifold_architectures.md`
- **Upstream artifacts consumed:** current-code spline/flow/path artifacts and VAE comparison-ready artifacts.
- **Downstream artifacts produced:** `${experiment_root}/compare_manifold_architectures/summary.csv`, `ablation_matrix.md`, `metric_deltas.json`, and figures.
- **Non-default knobs:** metric list and baseline arm names.
- **Pre-flight check:** every compared arm must identify its task, layer/token site, architecture, loss set, metric, and seed.
- **Estimated runtime + GPU footprint:** CPU/lightweight aggregation after artifacts exist.

### Cross-analysis post-steps

- Generate a single comparison table keyed by task, architecture arm, metric choice, loss set, and activation site.
- Copy the most diagnostic figures into `${SESSION_DIR}/result/figures/` during interpretation.
- In the final report, keep a separate section for "current-code reproduction" so VAE gains are not confused with missing baseline artifacts.

## Section E. Risk register and contingency

### Pitfalls active for this plan

- `task.resample_variable` and `locate.mode` mismatch is avoided by relying on current runner patterns and centroid-style comparisons, but should be revisited if a pairwise locate stage is added.
- No global `artifacts/` directory was found; current results must be rerun or supplied from a prior session.
- VAE patch losses can dominate runtime; the first sweep should train with behavior-head losses and run patch evaluation on selected arms/pairs.
- Atlas models can appear better merely because they have more parameters; compare against capacity-matched single-chart variants where possible.
- Known topology arms (`S1`, `R2`, cylinder) risk leaking task structure; report them separately from unstructured-latent arms.

### Per-step contingency

| Node | If pre-flight fails, then |
|---|---|
| baseline | Stop and fix task/tokenization/model accuracy before any manifold comparison. |
| subspace | Reduce to a known-good layer/token or rerun with a smaller debug dataset. |
| activation_manifold | Try `intrinsic_mode: parameter` for structured tasks or raise smoothness from 0.0. |
| output_manifold | Fall back to baseline distribution distances for VAE behavior losses, but skip pullback claims. |
| path_steering | Reduce `selected_pairs`, `n_prompts`, and path steps; do not launch full ablations until debug pairs work. |
| behavior_aligned_vae | Start with flat VAE + reconstruction/KL only, then add behavior/isometry losses one at a time. |
| behavior_manifold_vae | Disable patch evaluation and validate behavior-head metrics first. |
| compare_manifold_architectures | Emit missing-input rows rather than mixing session-local and unknown global artifacts. |

## Section F. Outputs of the plan itself

### Runner config(s)

Names only; `/run-experiment` should materialize the YAML under the session-local config tree.

- `${SESSION_DIR}/code/configs/runners/manifold_vae/weekdays_current_spline.yaml`
- `${SESSION_DIR}/code/configs/runners/manifold_vae/weekdays_vae_debug.yaml`
- `${SESSION_DIR}/code/configs/runners/manifold_vae/weekdays_vae_priority_sweep.yaml`
- `${SESSION_DIR}/code/configs/runners/manifold_vae/grid_current_spline.yaml`
- `${SESSION_DIR}/code/configs/runners/manifold_vae/grid_vae_priority_sweep.yaml`
- `${SESSION_DIR}/code/configs/runners/manifold_vae/compare_architectures.yaml`

### Sweep and cache strategy

- **Sweep ID:** `manifold-vae-priority`
- **Resolved root:** `${SESSION_DIR}/artifacts/{task}/{model}/manifold-vae-priority/`
- **Sweep axes:**
  - architecture: current spline, current flow optional, flat VAE, metric VAE, atlas VAE;
  - loss set: reconstruction-only vs behavior-aligned;
  - metric: latent-linear, decoder-pullback, behavior-pullback;
  - topology: unstructured vs known topology for the toy-domain sanity check.
- **Cache reuse plan:**
  - `baseline`, `subspace`, and `output_manifold` are shared per task/model/sweep root.
  - Current `activation_manifold` arms write separate `_subdir`s by method/smoothness.
  - VAE arms must include architecture, loss set, metric, topology, chart count, and seed in `_subdir`.
  - `path_steering` and VAE path metrics should share selected pair definitions for comparable held-out evaluation.
- **Overwrite hazards verified absent:**
  - Do not vary `task.target_variable` within one sweep root.
  - Do not let multiple VAE arms share the same `_subdir`.
  - Do not rerun current-code baselines into an older global path.

### Expected artifact tree

```text
${SESSION_DIR}/artifacts/{task}/{model}/manifold-vae-priority/
├── baseline/
│   ├── accuracy.json
│   ├── per_class_output_dists.safetensors
│   └── counterfactual_sanity.json
├── subspace/
│   └── ...
├── activation_manifold/
│   ├── spline_s0.0/
│   └── flow_.../                 # optional current-code flow arm
├── output_manifold/
│   └── spline_s0.0/
├── path_steering/
│   └── ...
├── behavior_manifold_vae/
│   ├── flat_recon_metric-latent_topo-unstructured_seed0/
│   ├── metric_beh_metric-decoder_topo-known_seed0/
│   ├── metric_beh_metric-behavior_topo-known_seed0/
│   └── atlas_beh_metric-behavior_chartsK_topo-known_seed0/
└── compare_manifold_architectures/
    ├── summary.csv
    ├── ablation_matrix.md
    ├── metric_deltas.json
    └── figures/
```

### Hand-off

- After approval, run `/setup-methods` for `setup_method_behavior_aligned_vae.md`.
- Then run `/setup-analyses` for `setup_analysis_behavior_manifold_vae.md` and `setup_analysis_compare_manifold_architectures.md`.
- Then run `/run-experiment` to materialize runner configs and execute debug current-code and VAE arms.
- Then run `/interpret-experiment` to produce `${SESSION_DIR}/result/REPORT.md`.

## Review checkpoint

Please review:

1. **Hypotheses + success criteria:** Do the criteria in `RESEARCH_OBJECTIVE.md` capture the comparison you want, especially the rule that VAE improvements must be behavior/path gains without severe reconstruction or patch-naturalness regressions?
2. **Sweep strategy:** Is the proposed cache-sharing sweep acceptable, or should current-code reproduction and VAE ablations use isolated experiment roots?
3. **Compute estimate:** Is the staged plan acceptable: weekdays debug first, weekdays priority sweep second, graph-walk after the VAE is stable?
