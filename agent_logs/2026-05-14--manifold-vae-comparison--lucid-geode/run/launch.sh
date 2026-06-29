#!/usr/bin/env bash
# Launch the weekdays manifold-VAE comparison with ALL outputs routed to the
# data directory (/data/jiang/vennemdp/causalab), not the repo tree.
#
# Run from the repo root on the GPU box:  bash agent_logs/<session>/run/launch.sh
#
# How the routing works:
#   - DATA_ROOT          : where every generated artifact lands.
#   - --experiment-root  : points under DATA_ROOT (Hydra's single output knob,
#                          ARCHITECTURE.md invariant 7). Because this is NOT under
#                          agent_logs/, we must tell the wrapper where the
#                          session-local VAE code lives via CAUSALAB_SESSION_CODE.
#   - CAUSALAB_SESSION_CODE : absolute path to the session dir (the one holding
#                          code/). Overrides the wrapper's path-based detection so
#                          `import analyses.behavior_manifold_vae` + the
#                          session-local runner configs still resolve.
#   - HF_HOME            : model weights cache also under DATA_ROOT (set only if
#                          you haven't already pointed it elsewhere).
set -euo pipefail

# --- knobs (override via env) -----------------------------------------------
DATA_ROOT="${DATA_ROOT:-/data/jiang/vennemdp/causalab}"
SESSION="${SESSION:-2026-05-14--manifold-vae-comparison--lucid-geode}"
MODEL="${MODEL:-llama31_8b}"          # e.g. MODEL=llama32_1b_instruct for a small cached model
TASK_DIRNAME="natural_domains_arithmetic_weekdays"
DEBUG="${DEBUG:-0}"                    # DEBUG=1 -> tiny dataset smoke pass for step 1

# --- resolve paths ----------------------------------------------------------
REPO_ROOT="$(pwd)"
SESSION_DIR="${REPO_ROOT}/agent_logs/${SESSION}"
[ -d "${SESSION_DIR}/code" ] || { echo "Session code not found at ${SESSION_DIR}/code — run from the repo root." >&2; exit 1; }

export CAUSALAB_SESSION_CODE="${SESSION_DIR}"
# Route the HuggingFace cache to DATA_ROOT too (only if the user hasn't set one).
: "${HF_HOME:=${DATA_ROOT}/hf_cache}"
export HF_HOME
mkdir -p "${HF_HOME}"

# scripts/run_exp.sh uses `uv run`, which builds the project's (large, CUDA)
# environment. By default uv writes its cache to ~/.cache/uv and the venv to
# <repo>/.venv — both on the HOME partition, which on this cluster is small and
# fills up ("No space left on device"). Redirect both to DATA_ROOT. Only set if
# the user hasn't already pointed them elsewhere.
: "${UV_CACHE_DIR:=${DATA_ROOT}/uv_cache}"
export UV_CACHE_DIR
: "${UV_PROJECT_ENVIRONMENT:=${DATA_ROOT}/uv_venv}"
export UV_PROJECT_ENVIRONMENT
mkdir -p "${UV_CACHE_DIR}"

DATA_ARTIFACTS="${DATA_ROOT}/${SESSION}/artifacts"
EXP_ROOT="${DATA_ARTIFACTS}/${TASK_DIRNAME}/${MODEL}"
LOG_DIR="${SESSION_DIR}/run"
mkdir -p "${EXP_ROOT}" "${LOG_DIR}"

# Keep the session self-contained without copying bytes: symlink the in-repo
# artifacts/ path to the data directory. This is what makes /interpret-experiment
# (which reads ${SESSION_DIR}/artifacts/...) find the outputs that physically
# live on DATA_ROOT. Skip if a real (non-symlink) artifacts dir already exists.
if [ -e "${SESSION_DIR}/artifacts" ] && [ ! -L "${SESSION_DIR}/artifacts" ]; then
  echo "WARNING: ${SESSION_DIR}/artifacts exists and is not a symlink — leaving it as-is." >&2
else
  ln -sfn "${DATA_ARTIFACTS}" "${SESSION_DIR}/artifacts"
  echo "symlink: ${SESSION_DIR}/artifacts -> ${DATA_ARTIFACTS}"
fi

echo "DATA_ROOT            = ${DATA_ROOT}"
echo "EXP_ROOT            = ${EXP_ROOT}"
echo "CAUSALAB_SESSION_CODE = ${CAUSALAB_SESSION_CODE}"
echo "HF_HOME             = ${HF_HOME}"
echo "UV_CACHE_DIR        = ${UV_CACHE_DIR}"
echo "UV_PROJECT_ENVIRONMENT = ${UV_PROJECT_ENVIRONMENT}"
echo "MODEL               = ${MODEL}   DEBUG=${DEBUG}"
echo

run() {  # run <runner> [extra hydra overrides...]
  local runner="$1"; shift
  echo ">>> ${runner} ${*:-}"
  ./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" "${runner}" model="${MODEL}" "$@" \
      > "${LOG_DIR}/run_${runner}.log" 2>&1
  echo "    done -> ${LOG_DIR}/run_${runner}.log"
}

# 1) Current-code reproduction (spline pipeline; produces the shared cache).
if [ "${DEBUG}" = "1" ]; then
  run weekdays_current_spline task.n_train=64 task.n_test=16
else
  run weekdays_current_spline
fi

# --- pre-flight gate: stop if the model can't do the task ---------------------
ACC="${EXP_ROOT}/baseline/accuracy.json"
echo
echo "Baseline accuracy ($ACC):"
[ -f "${ACC}" ] && cat "${ACC}" || echo "  (missing — check ${LOG_DIR}/run_weekdays_current_spline.log)"
echo "If weekday accuracy is poor, STOP and fix task/model before trusting the comparison."
echo

# 2) Debug VAE arm (reuses the shared cache).
run weekdays_vae_debug

# 3) Aggregate spline vs VAE.
run compare_architectures

echo
echo "All steps done. Artifacts under: ${EXP_ROOT}"
echo "Next: run /interpret-experiment (it reads agent_logs/.current) to write result/REPORT.md."
echo "Note: REPORT/figures read artifacts from DATA_ROOT; keep ${DATA_ROOT} mounted when interpreting."
