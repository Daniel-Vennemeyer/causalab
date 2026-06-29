# Issues

- **[ERROR]** Search command included a non-existent top-level `configs/` path
  - Context: Looking for existing manifold, spline, path-steering, and pullback code.
  - The repo uses `causalab/configs/`; passing `configs` to `rg` produced a harmless "No such file or directory" diagnostic.
  - Worked around it by reading the relevant files under `causalab/configs/analysis/` and `causalab/configs/runners/`.

- **[UNEXPECTED]** No global `artifacts/` directory was present in the checkout
  - Context: The user asked to compare fully against current-code results.
  - Without an existing artifact tree, the plan should either rerun current spline/flow baselines inside this session or reuse an explicitly named prior session.
  - The draft plan assumes fresh session-local reruns unless the user supplies an existing results path.

- **[BLOCKER → DEFERRED]** Execution environment does not match the plan's compute assumption (2026-06-18)
  - Plan targets `meta-llama/Llama-3.1-8B` on a CUDA GPU. Actual machine is a Mac: MPS only (no CUDA), 34 GB unified memory, and the gated 8B weights are NOT downloaded. Only `meta-llama/Llama-3.2-1B` (base) and `Llama-3.3-70B-Instruct` are in the HF cache.
  - **Decision:** defer execution to a GPU box. Method + analyses are built and validated; runner config(s) are materialized and dry-run-checked (`--cfg job`) but NOT executed here. Launch later on CUDA with the 8B model (inline or slurm).
  - Status as of deferral: `code/methods/behavior_aligned_vae` (27 tests green), `code/analyses/{behavior_manifold_vae,compare_manifold_architectures}` (import + YAML validated) complete.

- **[CONFIG]** Outputs routed to the data directory `/data/jiang/vennemdp/causalab` (2026-06-18)
  - Per user request, all generated artifacts + the HF model cache are routed to `$DATA_ROOT`, not the repo tree. Mechanism: `--experiment-root` under `$DATA_ROOT` + `CAUSALAB_SESSION_CODE=<abs session dir>` (so the wrapper still injects session-local VAE code even though the root is outside `agent_logs/`) + `HF_HOME=$DATA_ROOT/hf_cache`.
  - `run/launch.sh` wires this up and symlinks `agent_logs/<session>/artifacts` → `$DATA_ROOT/<session>/artifacts` so the session stays self-contained and `/interpret-experiment` finds the outputs.
  - Verified on the Mac via `--cfg job` with an out-of-tree root: session code injects and `experiment_root` resolves to the data path.
