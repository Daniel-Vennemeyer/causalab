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
TASK_DIRNAME="natural_domains_arithmetic_weekdays"

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

fail=0
for arm in "${ARMS[@]}"; do
  echo ">>> ${arm} ${PATCH_OVERRIDE}"
  if ./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" "${arm}" model="${MODEL}" ${PATCH_OVERRIDE:+$PATCH_OVERRIDE} \
        > "${LOG_DIR}/run_${arm}.log" 2>&1; then
    echo "    done -> ${LOG_DIR}/run_${arm}.log"
  else
    echo "    FAILED -> ${LOG_DIR}/run_${arm}.log (continuing)"
    tail -15 "${LOG_DIR}/run_${arm}.log" | sed 's/^/      /'
    fail=1
  fi
done

echo
echo ">>> compare_architectures"
./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" compare_architectures model="${MODEL}" \
    > "${LOG_DIR}/run_compare_architectures.log" 2>&1
echo "    done -> ${LOG_DIR}/run_compare_architectures.log"

echo
echo "===== summary.csv ====="
find "${EXP_ROOT}" -name summary.csv -exec cat {} \;
echo
echo "===== metric_deltas.json ====="
find "${EXP_ROOT}" -name metric_deltas.json -exec cat {} \;
[ "${fail}" = "0" ] || echo "(one or more arms failed — see logs above)"
