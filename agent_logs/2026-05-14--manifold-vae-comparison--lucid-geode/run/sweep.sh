#!/usr/bin/env bash
# Priority sweep for the manifold-VAE comparison: 5 behavior-aligned VAE arms
# (Ablations 2/3/4a/4b/6), then re-aggregate with compare_architectures.
#
# Reuses the shared-root spline cache from weekdays_current_spline (baseline,
# subspace, output_manifold) — no model load, CPU VAE training, minutes total.
# Run from the repo root on the GPU box (CUDA not actually needed here):
#   bash agent_logs/<session>/run/sweep.sh
#
# Same /data routing as launch.sh (see that file for the rationale).
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/jiang/vennemdp/causalab}"
SESSION="${SESSION:-2026-05-14--manifold-vae-comparison--lucid-geode}"
MODEL="${MODEL:-llama31_8b}"
# Per-domain EXP_ROOT (caches keyed by this). All natural_domains variants share
# task.name=natural_domains_arithmetic, so distinct dirs are REQUIRED to avoid
# cross-domain cache collisions. Override per domain, e.g.
# TASK_DIRNAME=natural_domains_arithmetic_months.
TASK_DIRNAME="${TASK_DIRNAME:-natural_domains_arithmetic_weekdays}"

REPO_ROOT="$(pwd)"
SESSION_DIR="${REPO_ROOT}/agent_logs/${SESSION}"
[ -d "${SESSION_DIR}/code" ] || { echo "Run from the repo root (need ${SESSION_DIR}/code)." >&2; exit 1; }

export CAUSALAB_SESSION_CODE="${SESSION_DIR}"
: "${HF_HOME:=${DATA_ROOT}/hf_cache}"; export HF_HOME
: "${UV_CACHE_DIR:=${DATA_ROOT}/uv_cache}"; export UV_CACHE_DIR
: "${UV_PROJECT_ENVIRONMENT:=${DATA_ROOT}/uv_venv}"; export UV_PROJECT_ENVIRONMENT

EXP_ROOT="${DATA_ROOT}/${SESSION}/artifacts/${TASK_DIRNAME}/${MODEL}"
LOG_DIR="${SESSION_DIR}/run"
mkdir -p "${LOG_DIR}"
ln -sfn "${DATA_ROOT}/${SESSION}/artifacts" "${SESSION_DIR}/artifacts" 2>/dev/null || true

echo "EXP_ROOT = ${EXP_ROOT}"
echo "MODEL    = ${MODEL}"
echo

# All 5 arms by default; override e.g. ARMS="weekdays_vae_aligned_flat" to run one.
read -r -a ARMS <<< "${ARMS:-weekdays_vae_aligned_flat weekdays_vae_aligned_decoder_metric weekdays_vae_aligned_behavior_metric weekdays_vae_aligned_atlas weekdays_vae_aligned_s1}"

# PATCH=1 re-runs each arm with patch_eval=true: re-trains the (cheap) VAE, then
# patches its decoded steered paths into the frozen LM and scores coherence /
# distance_from_behavior_manifold on the SAME axes as the spline. This LOADS THE
# 8B MODEL and is the GPU-heavy path. Default off (geometry-proxy metrics only).
PATCH_OVERRIDE=""
if [ "${PATCH:-0}" = "1" ]; then
  PATCH_OVERRIDE="behavior_manifold_vae.patch_eval=true"
  echo "PATCH=1 -> patch_eval=true (loads the 8B model)"
fi

# Seeds to run per arm (space-separated). Default 42 (matches base.yaml).
# e.g. SEEDS="0 1 2 3 4" for error bars. Each (arm, seed) writes a distinct
# _seed{seed} dir, so seeds accumulate without clobbering.
read -r -a SEEDS_ARR <<< "${SEEDS:-42}"
echo "SEEDS = ${SEEDS_ARR[*]}"

# Device for VAE training + geodesic/metric compute. NOTE: the dominant cost of
# the non-patch sweep is per-run process startup (importing the stack), not the
# tiny VAE (≈1s train, ≈10s for the 21-geodesic metric arms). GPU barely helps
# those launch-bound ops — JOBS (parallelism) is the real speedup. cuda is the
# default to honor the GPU box; set DEVICE=cpu if CUDA contention is an issue.
DEVICE="${DEVICE:-cuda}"

# JOBS = how many (arm,seed) runs to run concurrently. The runs are independent
# (distinct output dirs, shared read-only cache), so parallelism hides the
# import/startup latency that dominates wall-clock. PATCH=1 loads the 8B model
# per run, so it is forced serial (multiple 8B models would OOM the GPU).
JOBS="${JOBS:-4}"
if [ "${PATCH:-0}" = "1" ]; then JOBS=1; fi

# CRITICAL for parallel CPU runs: PyTorch/BLAS default to ALL cores per process,
# so JOBS parallel workers each grab every core → hundreds of threads thrash the
# node and a 2s job takes 30+ min. Cap threads PER worker so total ≈ THREADS×JOBS
# stays sane. 1 thread/worker is plenty here (tiny tensors); raise THREADS if you
# lower JOBS. Honors a pre-set value if you exported one.
THREADS="${THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$THREADS}"
export TOKENIZERS_PARALLELISM=false

# Optional: cap geodesic-solver iterations (the metric arms' dominant compute —
# ~10s at the config default of 100). GEO_ITERS=40 ≈ 2.5x faster for those arms
# with negligible path-quality loss on a 2-D latent. Empty = use the config.
GEO_OVERRIDE=""
if [ -n "${GEO_ITERS:-}" ]; then
  GEO_OVERRIDE="behavior_manifold_vae.geodesic.n_iters=${GEO_ITERS}"
fi

n_runs=$(( ${#ARMS[@]} * ${#SEEDS_ARR[@]} ))
echo "DEVICE = ${DEVICE}   JOBS = ${JOBS}   THREADS/worker = ${THREADS}   runs = ${n_runs} (${#ARMS[@]} arms × ${#SEEDS_ARR[@]} seeds)"
[ -n "${GEO_OVERRIDE}" ] && echo "GEO override: ${GEO_OVERRIDE}"
echo

# One run. Always returns 0 (failures are logged, not fatal) so a single bad arm
# doesn't abort the whole sweep. Exported for the xargs workers below.
run_one() {  # run_one <arm> <seed>
  local arm="$1" seed="$2"
  local log="${LOG_DIR}/run_${arm}_seed${seed}.log"
  local rc=0
  echo ">>> ${arm} seed=${seed}"
  if [ "${JOBS}" = "1" ]; then
    # serial → stream live so the in-run tqdm progress bars are visible, tee to log
    ./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" "${arm}" \
      model="${MODEL}" seed="${seed}" behavior_manifold_vae.device="${DEVICE}" \
      ${PATCH_OVERRIDE:+$PATCH_OVERRIDE} ${GEO_OVERRIDE:+$GEO_OVERRIDE} 2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
  else
    # parallel → quiet to log (interleaved live bars would be unreadable)
    ./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" "${arm}" \
      model="${MODEL}" seed="${seed}" behavior_manifold_vae.device="${DEVICE}" \
      ${PATCH_OVERRIDE:+$PATCH_OVERRIDE} ${GEO_OVERRIDE:+$GEO_OVERRIDE} > "${log}" 2>&1 || rc=$?
  fi
  if [ "${rc}" = "0" ]; then
    echo "    done -> ${log}"
  else
    echo "    FAILED -> ${log} (see log)"
    tail -8 "${log}" | sed 's/^/      /'
  fi
}
export -f run_one
export EXP_ROOT MODEL DEVICE LOG_DIR PATCH_OVERRIDE GEO_OVERRIDE JOBS

# Run all (arm, seed) combos with up to JOBS concurrent workers (xargs -P is
# portable across bash versions; runs are independent so this just hides the
# per-run import/startup latency that dominates wall-clock).
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS_ARR[@]}"; do
    printf '%s %s\n' "${arm}" "${seed}"
  done
done | xargs -P "${JOBS}" -L1 bash -c 'run_one "$@"' _

echo
echo ">>> compare_architectures"
./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" compare_architectures model="${MODEL}" \
    > "${LOG_DIR}/run_compare_architectures.log" 2>&1
echo "    done -> ${LOG_DIR}/run_compare_architectures.log"

echo
echo "===== summary_by_arm.csv (mean±std across seeds) ====="
find "${EXP_ROOT}" -name summary_by_arm.csv -exec cat {} \;
echo
echo "===== summary.csv (per arm/seed) ====="
find "${EXP_ROOT}" -name summary.csv -exec cat {} \;
echo
echo "===== metric_deltas.json ====="
find "${EXP_ROOT}" -name metric_deltas.json -exec cat {} \;
echo
echo "(check run/*.log for any arms that failed; grep -L 'comparison_ready' won't apply — inspect logs)"
