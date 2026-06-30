#!/usr/bin/env bash
# Phase-1 generalization: replicate the weekdays manifold-VAE result on the other
# natural_domains arithmetic domains.
#
#   months   (cyclic, W=12) : spline + transition_centroid + transition_manifold, PATCH
#   alphabet (line,   W=22) : spline + transition_centroid + transition_manifold, PATCH
#   age      (line,   W=91) : subspace-only + transition_centroid, DISCOVERY-ONLY (no patch)
#
# Each domain gets its OWN EXP_ROOT (task.name collides across domains, so caches
# must be separated). Run from the repo root on the GPU box:
#   bash agent_logs/<session>/run/launch_generalization.sh
# Override the domain list:  DOMAINS="months" bash .../launch_generalization.sh
#
# Same /data routing rationale as launch.sh.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/jiang/vennemdp/causalab}"
SESSION="${SESSION:-2026-05-14--manifold-vae-comparison--lucid-geode}"
MODEL="${MODEL:-llama31_8b}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-cpu}"
CUDA="${CUDA_VISIBLE_DEVICES:-3}"

REPO_ROOT="$(pwd)"
SESSION_DIR="${REPO_ROOT}/agent_logs/${SESSION}"
[ -d "${SESSION_DIR}/code" ] || { echo "Run from the repo root (need ${SESSION_DIR}/code)." >&2; exit 1; }

export CAUSALAB_SESSION_CODE="${SESSION_DIR}"
: "${HF_HOME:=${DATA_ROOT}/hf_cache}"; export HF_HOME
: "${UV_CACHE_DIR:=${DATA_ROOT}/uv_cache}"; export UV_CACHE_DIR
: "${UV_PROJECT_ENVIRONMENT:=${DATA_ROOT}/uv_venv}"; export UV_PROJECT_ENVIRONMENT
LOG_DIR="${SESSION_DIR}/run"; mkdir -p "${LOG_DIR}"

# domain -> "task_dirname | spline_runner | patch | arms"
declare -A SPEC
SPEC[months]="natural_domains_arithmetic_months|months_current_spline|1|months_vae_transition_centroid months_vae_transition_manifold"
SPEC[alphabet]="natural_domains_arithmetic_alphabet|alphabet_current_spline|1|alphabet_vae_transition_centroid alphabet_vae_transition_manifold"
SPEC[age]="natural_domains_arithmetic_age|age_discovery_baseline|0|age_vae_transition_centroid"

DOMAINS="${DOMAINS:-months alphabet age}"

for dom in ${DOMAINS}; do
  IFS='|' read -r TDIR SPLINE PATCHFLAG ARMS <<< "${SPEC[$dom]}"
  EXP_ROOT="${DATA_ROOT}/${SESSION}/artifacts/${TDIR}/${MODEL}"
  mkdir -p "${EXP_ROOT}"
  echo "============================================================"
  echo "DOMAIN=${dom}  EXP_ROOT=${EXP_ROOT}  patch=${PATCHFLAG}"
  echo "============================================================"

  # 1) Baseline / spline pipeline (loads the 8B model; produces subspace cache).
  echo ">>> baseline runner: ${SPLINE}"
  ./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" "${SPLINE}" model="${MODEL}" \
      > "${LOG_DIR}/gen_${dom}_${SPLINE}.log" 2>&1 \
      && echo "    done -> ${LOG_DIR}/gen_${dom}_${SPLINE}.log" \
      || { echo "    FAILED -> ${LOG_DIR}/gen_${dom}_${SPLINE}.log"; tail -8 "${LOG_DIR}/gen_${dom}_${SPLINE}.log" | sed 's/^/      /'; continue; }

  # 2) VAE arms via the shared sweep (handles thread caps, seeds, compare).
  echo ">>> arms: ${ARMS}  (PATCH=${PATCHFLAG})"
  TASK_DIRNAME="${TDIR}" PATCH="${PATCHFLAG}" CUDA_VISIBLE_DEVICES="${CUDA}" \
    DEVICE="${DEVICE}" ARMS="${ARMS}" SEEDS="${SEEDS}" \
    bash "${SESSION_DIR}/run/sweep.sh"
done

echo
echo "All domains done. Per-domain summaries:"
for dom in ${DOMAINS}; do
  IFS='|' read -r TDIR _ _ _ <<< "${SPEC[$dom]}"
  echo "  ${dom}: ${DATA_ROOT}/${SESSION}/artifacts/${TDIR}/${MODEL}/**/summary_by_arm.csv"
done
