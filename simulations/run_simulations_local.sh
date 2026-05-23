#!/bin/bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_simulations_local.sh --config <config_name_or_path> [options]
  ./run_simulations_local.sh --all-configs [options]

Core options:
  --config <config>            Single config to run
  --all-configs                Run all simulations/config_*.py in sorted order
  --mode <auto|np|sp>          Runner mode (default: auto)
  --node-id <int>              Placeholder replacement for __NODEID__ (default: 0)
  --n-nodes <int>              Placeholder replacement for __NNODES__ (default: 1)
  --python <path>              Python executable to use (default: python)
  --timing-log <path>          Optional CSV path to append per-config runtimes

Runtime overrides:
  --n-experiments <int>
  --seed <int>
  --smoke-test
  --force-rerun
  --smoke-n-experiments <int>  (default: 5)
  --smoke-max-n-samples <int>
  --smoke-max-n-test <int>
  --smoke-max-n-epochs <int>   (default: 150)
  --smoke-max-burnin <int>     (default: 100)

Use "--" to pass any additional arguments directly to sweep_np.py/sweep_sp.py.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="auto"
CONFIG=""
ALL_CONFIGS=0
NODE_ID=0
N_NODES=1
PYTHON_BIN="${PYTHON:-python}"
TIMING_LOG=""

N_EXPERIMENTS=""
SEED=""
SMOKE_TEST=0
FORCE_RERUN=0
SMOKE_N_EXPERIMENTS=5
SMOKE_MAX_N_SAMPLES=500
SMOKE_MAX_N_TEST=500
SMOKE_MAX_N_EPOCHS=150
SMOKE_MAX_BURNIN=100

EXTRA_SWEEP_ARGS=()
TIMING_CONFIGS=()
TIMING_MODES=()
TIMING_STATUSES=()
TIMING_SECONDS=()
TIMING_SUMMARY_PRINTED=0
SCRIPT_START_TS="$(date +%s)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --all-configs)
      ALL_CONFIGS=1
      shift
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --node-id)
      NODE_ID="$2"
      shift 2
      ;;
    --n-nodes)
      N_NODES="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --timing-log)
      TIMING_LOG="$2"
      shift 2
      ;;
    --n-experiments)
      N_EXPERIMENTS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --smoke-test)
      SMOKE_TEST=1
      shift
      ;;
    --force-rerun)
      FORCE_RERUN=1
      shift
      ;;
    --smoke-n-experiments)
      SMOKE_N_EXPERIMENTS="$2"
      shift 2
      ;;
    --smoke-max-n-samples)
      SMOKE_MAX_N_SAMPLES="$2"
      shift 2
      ;;
    --smoke-max-n-test)
      SMOKE_MAX_N_TEST="$2"
      shift 2
      ;;
    --smoke-max-n-epochs)
      SMOKE_MAX_N_EPOCHS="$2"
      shift 2
      ;;
    --smoke-max-burnin)
      SMOKE_MAX_BURNIN="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_SWEEP_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ${ALL_CONFIGS} -eq 1 && -n "${CONFIG}" ]]; then
  echo "Use either --config or --all-configs, not both." >&2
  exit 1
fi
if [[ ${ALL_CONFIGS} -eq 0 && -z "${CONFIG}" ]]; then
  echo "You must provide one of --config or --all-configs." >&2
  exit 1
fi
if [[ "${MODE}" != "auto" && "${MODE}" != "np" && "${MODE}" != "sp" ]]; then
  echo "Invalid --mode ${MODE}; expected auto|np|sp." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

format_duration() {
  local total_seconds="$1"
  local hours=$(( total_seconds / 3600 ))
  local minutes=$(( (total_seconds % 3600) / 60 ))
  local seconds=$(( total_seconds % 60 ))
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

record_timing() {
  local config_name="$1"
  local run_mode="$2"
  local status_text="$3"
  local elapsed_seconds="$4"

  TIMING_CONFIGS+=("${config_name}")
  TIMING_MODES+=("${run_mode}")
  TIMING_STATUSES+=("${status_text}")
  TIMING_SECONDS+=("${elapsed_seconds}")

  if [[ -n "${TIMING_LOG}" ]]; then
    printf "%s,%s,%s,%s,%s,%s\n" \
      "$(date +%Y-%m-%dT%H:%M:%S%z)" \
      "${config_name}" \
      "${run_mode}" \
      "${status_text}" \
      "${elapsed_seconds}" \
      "$(format_duration "${elapsed_seconds}")" >> "${TIMING_LOG}"
  fi
}

print_timing_summary() {
  if [[ "${TIMING_SUMMARY_PRINTED}" -eq 1 ]]; then
    return
  fi
  TIMING_SUMMARY_PRINTED=1

  local count="${#TIMING_CONFIGS[@]}"
  if [[ "${count}" -eq 0 ]]; then
    return
  fi

  local ok_count=0
  local fail_count=0
  local elapsed_total=0
  local i

  echo
  echo "Timing summary:"
  for (( i=0; i<count; i++ )); do
    local cfg="${TIMING_CONFIGS[$i]}"
    local mode="${TIMING_MODES[$i]}"
    local status="${TIMING_STATUSES[$i]}"
    local seconds="${TIMING_SECONDS[$i]}"
    local hhmmss
    hhmmss="$(format_duration "${seconds}")"

    echo "  - ${cfg} (mode=${mode}): ${hhmmss} (${seconds}s) [${status}]"

    elapsed_total=$(( elapsed_total + seconds ))
    if [[ "${status}" == "ok" ]]; then
      ok_count=$(( ok_count + 1 ))
    else
      fail_count=$(( fail_count + 1 ))
    fi
  done

  local script_total_seconds
  script_total_seconds=$(( $(date +%s) - SCRIPT_START_TS ))
  echo "Total configs: ${count} | ok: ${ok_count} | failed: ${fail_count}"
  echo "Sum of config runtimes: $(format_duration "${elapsed_total}") (${elapsed_total}s)"
  echo "Wall-clock elapsed: $(format_duration "${script_total_seconds}") (${script_total_seconds}s)"
}

if [[ -n "${TIMING_LOG}" ]]; then
  mkdir -p "$(dirname "${TIMING_LOG}")"
  if [[ ! -f "${TIMING_LOG}" ]]; then
    echo "timestamp,config,mode,status,seconds,hhmmss" > "${TIMING_LOG}"
  fi
fi

trap print_timing_summary EXIT

# Avoid oversubscription: cap BLAS/OpenMP threads to 1 per process
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

resolve_config_path() {
  local raw="$1"
  local trimmed="${raw%.py}"
  trimmed="${trimmed#./}"

  local dotted
  dotted="$(echo "${trimmed}" | tr '.' '/')"

  local candidates=(
    "${SCRIPT_DIR}/${trimmed}.py"
    "${SCRIPT_DIR}/${trimmed}"
    "${SCRIPT_DIR}/${dotted}.py"
    "${REPO_ROOT}/${trimmed}.py"
    "${REPO_ROOT}/${trimmed}"
    "${REPO_ROOT}/${dotted}.py"
  )

  local c
  for c in "${candidates[@]}"; do
    if [[ -f "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done

  return 1
}

run_one_config() {
  local config_input="$1"
  local config_path
  config_path="$(resolve_config_path "${config_input}")" || {
    echo "Could not resolve config from input: ${config_input}" >&2
    return 1
  }

  local config_basename
  config_basename="$(basename "${config_path}" .py)"

  local run_mode="${MODE}"
  if [[ "${run_mode}" == "auto" ]]; then
    case "${config_basename}" in
      config_np_*)
        run_mode="np"
        ;;
      config_sp_*)
        run_mode="sp"
        ;;
      *)
        echo "Unable to infer mode for ${config_basename}. Use --mode np|sp." >&2
        return 1
        ;;
    esac
  fi

  local runner
  if [[ "${run_mode}" == "np" ]]; then
    runner="sweep_np.py"
  else
    runner="sweep_sp.py"
  fi

  local tmp_dir="${SCRIPT_DIR}/temp"
  local tmp_file="${NODE_ID}_${config_basename}"
  local tmp_path="${tmp_dir}/${tmp_file}.py"
  mkdir -p "${tmp_dir}"
  cp "${config_path}" "${tmp_path}"

  sed -i.bak "s/__NODEID__/${NODE_ID}/g" "${tmp_path}"
  sed -i.bak "s/__NNODES__/${N_NODES}/g" "${tmp_path}"
  rm -f "${tmp_path}.bak"

  local runner_args=(--config "temp.${tmp_file}")

  if [[ -n "${N_EXPERIMENTS}" ]]; then
    runner_args+=(--n-experiments "${N_EXPERIMENTS}")
  fi
  if [[ -n "${SEED}" ]]; then
    runner_args+=(--seed "${SEED}")
  fi
  if [[ "${SMOKE_TEST}" -eq 1 ]]; then
    runner_args+=(
      --smoke-test
      --smoke-n-experiments "${SMOKE_N_EXPERIMENTS}"
      --smoke-max-n-samples "${SMOKE_MAX_N_SAMPLES}"
      --smoke-max-n-test "${SMOKE_MAX_N_TEST}"
      --smoke-max-n-epochs "${SMOKE_MAX_N_EPOCHS}"
      --smoke-max-burnin "${SMOKE_MAX_BURNIN}"
    )
  fi
  if [[ "${FORCE_RERUN}" -eq 1 ]]; then
    runner_args+=(--force-rerun)
  fi
  if [[ ${#EXTRA_SWEEP_ARGS[@]} -gt 0 ]]; then
    runner_args+=("${EXTRA_SWEEP_ARGS[@]}")
  fi

  local start_ts
  local end_ts
  local elapsed_seconds
  local status_code
  local status_text

  start_ts="$(date +%s)"
  echo "Running ${config_basename} (mode=${run_mode})"
  if (
    cd "${SCRIPT_DIR}"
    "${PYTHON_BIN}" "${runner}" "${runner_args[@]}"
  ); then
    status_code=0
    status_text="ok"
  else
    status_code=$?
    status_text="failed(${status_code})"
  fi
  end_ts="$(date +%s)"
  elapsed_seconds=$(( end_ts - start_ts ))

  rm -f "${tmp_path}"
  rm -f "${tmp_path}.bak"

  record_timing "${config_basename}" "${run_mode}" "${status_text}" "${elapsed_seconds}"
  echo "Finished ${config_basename} in $(format_duration "${elapsed_seconds}") (${elapsed_seconds}s) [${status_text}]"

  return "${status_code}"
}

if [[ ${ALL_CONFIGS} -eq 1 ]]; then
  configs=()
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    configs+=("${line%.py}")
  done < <(
    cd "${SCRIPT_DIR}"
    for f in config_*.py; do
      [[ -e "${f}" ]] || continue
      echo "${f}"
    done | sort
  )

  if [[ ${#configs[@]} -eq 0 ]]; then
    echo "No config_*.py files found in ${SCRIPT_DIR}." >&2
    exit 1
  fi

  for cfg in "${configs[@]}"; do
    run_one_config "${cfg}"
  done
else
  run_one_config "${CONFIG}"
fi
