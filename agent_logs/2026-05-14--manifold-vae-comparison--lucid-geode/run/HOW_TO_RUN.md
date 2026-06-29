# How to run the weekdays manifold-VAE comparison (deferred to a GPU box)

The method (`behavior_aligned_vae`) and analyses (`behavior_manifold_vae`,
`compare_manifold_architectures`) are built and tested; the three runner configs
below are validated via `--cfg job` (snapshots in this dir). Execution was
**deferred** because the dev machine is a Mac (MPS, no CUDA) and the gated 8B
weights aren't downloaded — see `../issues.md`.

## Prereqs on the GPU box
- CUDA GPU, and access to `meta-llama/Llama-3.1-8B` (HF auth for the gated repo),
  OR override `model=` with a model you have cached.
- Repo checked out at this session's commit; run from the repo root.

## Output routing → /data/jiang/vennemdp/causalab
ALL generated artifacts (and the HF model cache) are routed to the data directory,
NOT the repo tree. The single knob is `--experiment-root` (Hydra invariant 7);
because that path is outside `agent_logs/`, we set `CAUSALAB_SESSION_CODE` so the
wrapper still injects this session's VAE method + analyses. The session bundle
itself (plan/run/result/code — all small text) stays in the repo under
`agent_logs/`. `launch.sh` wires all of this up.

## Launch (one command — runs all three steps in order, shared cache)

```bash
# from the repo root on the GPU box:
bash agent_logs/2026-05-14--manifold-vae-comparison--lucid-geode/run/launch.sh
```

Overridable via env vars (defaults shown):
- `DATA_ROOT=/data/jiang/vennemdp/causalab`  — where everything is saved
- `MODEL=llama31_8b`                          — e.g. `MODEL=llama32_1b_instruct`
- `DEBUG=0`                                   — `DEBUG=1` runs a tiny smoke pass for step 1
- `HF_HOME` (defaults to `$DATA_ROOT/hf_cache`) — model weights cache

Example smoke pass with a small cached model:
```bash
DEBUG=1 MODEL=llama32_1b_instruct \
  bash agent_logs/2026-05-14--manifold-vae-comparison--lucid-geode/run/launch.sh
```

### Manual equivalent (if you'd rather run steps individually)
```bash
SESSION=2026-05-14--manifold-vae-comparison--lucid-geode
export CAUSALAB_SESSION_CODE="$(pwd)/agent_logs/$SESSION"
export HF_HOME=/data/jiang/vennemdp/causalab/hf_cache
EXP_ROOT="/data/jiang/vennemdp/causalab/$SESSION/artifacts/natural_domains_arithmetic_weekdays/llama31_8b"

./scripts/run_exp.sh --experiment-root "$EXP_ROOT" weekdays_current_spline
./scripts/run_exp.sh --experiment-root "$EXP_ROOT" weekdays_vae_debug
./scripts/run_exp.sh --experiment-root "$EXP_ROOT" compare_architectures
```
`--slurm` works too (resources resolve from the model config); the
`--experiment-root` and `CAUSALAB_SESSION_CODE` are forwarded into the sbatch step.

## Smaller-model override (no 8B / no CUDA)
Append `model=llama32_1b_instruct` (download ~2.5GB) to any command. Accuracy may
be lower; the baseline pre-flight gate (accuracy + counterfactual sanity) tells
you whether the run is meaningful. Add `behavior_manifold_vae.device=cuda` if you
want VAE training on GPU (it's tiny; cpu is fine).

## Pre-flight gate (STOP conditions)
- After step 1, check `$EXP_ROOT/baseline/accuracy.json` and
  `counterfactual_sanity.json`. If weekday accuracy is poor, stop and fix the
  task/model/tokenization before trusting any manifold comparison.

## Then interpret
`launch.sh` symlinks `agent_logs/<session>/artifacts` → `$DATA_ROOT/<session>/artifacts`,
so the session stays self-contained even though the bytes live on `/data`. Once
artifacts exist, run `/interpret-experiment` (it reads this session automatically
via that symlink) to produce `result/REPORT.md`. Keep `$DATA_ROOT` mounted while
interpreting.

## Expand to the priority sweep (after debug is clean)
Copy `weekdays_vae_debug.yaml` per arm, varying:
- `method`: flat_vae | metric_vae | atlas_vae (+ `n_charts`)
- `metric`: latent_linear | decoder_pullback | behavior_pullback
- `topology`: unstructured | s1 (weekdays known topology)
- `loss_weights`: recon-only vs behavior-aligned (`w_behavior`, `w_isometry` > 0)
Each arm auto-writes a distinct `_subdir` (method/topology/metric/charts/seed), so
they accumulate under the shared root without clobbering. Re-run
`compare_architectures` after the sweep to refresh the comparison.
```
