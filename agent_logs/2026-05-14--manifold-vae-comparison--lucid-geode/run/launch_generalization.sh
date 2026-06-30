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
DEVICE="${DEVICE:-cpu}"          # VAE/geodesic compute (tiny); the 8B model is device=auto -> GPU
CUDA="${CUDA_VISIBLE_DEVICES:-3}"
# GPU throughput knob: 8B forward batch for path_steering (baseline) AND the VAE
# patch eval. Bigger = fewer batches = better GPU util. 256 is a safe default on
# an 8B over short prompts; raise to 384/512 if the GPU has headroom, lower if OOM.
BATCH="${BATCH:-256}"

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

# count for the [i/N] progress banner
n_total=0; for _d in ${DOMAINS}; do n_total=$((n_total+1)); done
echo "Domains: ${DOMAINS}   GPU=${CUDA}   BATCH=${BATCH}   SEEDS=${SEEDS}"
echo

i_dom=0
for dom in ${DOMAINS}; do
  i_dom=$((i_dom+1))
  IFS='|' read -r TDIR SPLINE PATCHFLAG ARMS <<< "${SPEC[$dom]}"
  EXP_ROOT="${DATA_ROOT}/${SESSION}/artifacts/${TDIR}/${MODEL}"
  mkdir -p "${EXP_ROOT}"
  echo "============================================================"
  echo "[domain ${i_dom}/${n_total}] ${dom}   EXP_ROOT=${EXP_ROOT}   patch=${PATCHFLAG}"
  echo "============================================================"

  # path_steering exists only in the full spline runners (months/alphabet), not in
  # age_discovery_baseline — pass the GPU batch override only when applicable.
  STEER_OVERRIDE=""
  case "${SPLINE}" in *current_spline) STEER_OVERRIDE="path_steering.batch_size=${BATCH}";; esac

  # 1) Baseline / spline pipeline (loads the 8B model on GPU; produces the cache).
  #    Streamed live via `tee` so path_steering's tqdm bars are visible; GPU pinned.
  echo ">>> [${i_dom}/${n_total}] baseline runner: ${SPLINE}  (GPU=${CUDA})"
  if CUDA_VISIBLE_DEVICES="${CUDA}" ./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" \
        "${SPLINE}" model="${MODEL}" ${STEER_OVERRIDE:+$STEER_OVERRIDE} 2>&1 \
        | tee "${LOG_DIR}/gen_${dom}_${SPLINE}.log"; then
    echo "    done -> ${LOG_DIR}/gen_${dom}_${SPLINE}.log"
  else
    echo "    FAILED -> ${LOG_DIR}/gen_${dom}_${SPLINE}.log (see log)"; continue
  fi

  # 2) VAE arms via the shared sweep (handles thread caps, seeds, compare).
  #    PATCH=1 -> sweep forces JOBS=1 and streams live. For age (PATCH=0) force
  #    JOBS=1 too so its training/isometry tqdm streams instead of going quiet.
  JOBS_ENV=""; [ "${PATCHFLAG}" = "0" ] && JOBS_ENV="1"
  echo ">>> [${i_dom}/${n_total}] arms: ${ARMS}  (PATCH=${PATCHFLAG}, BATCH=${BATCH})"
  TASK_DIRNAME="${TDIR}" PATCH="${PATCHFLAG}" CUDA_VISIBLE_DEVICES="${CUDA}" \
    DEVICE="${DEVICE}" ARMS="${ARMS}" SEEDS="${SEEDS}" BATCH="${BATCH}" \
    ${JOBS_ENV:+JOBS=$JOBS_ENV} \
    bash "${SESSION_DIR}/run/sweep.sh"
done

echo
echo "All domains done. Per-domain summaries:"
for dom in ${DOMAINS}; do
  IFS='|' read -r TDIR _ _ _ <<< "${SPEC[$dom]}"
  echo "  ${dom}: ${DATA_ROOT}/${SESSION}/artifacts/${TDIR}/${MODEL}/**/summary_by_arm.csv"
done
