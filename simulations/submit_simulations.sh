#!/bin/bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./submit_simulations.sh --profile <profile> --config <config_name_or_path> [options]
  ./submit_simulations.sh --profile <profile> --all-configs [options]
  ./submit_simulations.sh --profile <profile> --all-np-configs [options]
  ./submit_simulations.sh --profile <profile> --all-sp-configs [options]

Profiles:
  sapphire | test | shared | unrestricted | intermediate

Core options:
  --profile <name>             Required Slurm resource profile
  --config <config>            Single config to run
  --all-configs                Submit all simulations/config_*.py as Slurm array jobs
  --all-np-configs             Submit all simulations/config_np_*.py as Slurm array jobs
  --all-sp-configs             Submit all simulations/config_sp_*.py as Slurm array jobs
  --mode <auto|np|sp>          Runner mode (default: auto)
  --node-id <int>              Placeholder replacement for __NODEID__ (default: 0)
  --n-nodes <int>              Placeholder replacement for __NNODES__ (default: 1)

Runtime overrides:
  --n-experiments <int>
  --seed <int>
  --smoke-test
  --force-rerun
  --smoke-n-experiments <int>
  --smoke-max-n-samples <int>
  --smoke-max-n-test <int>
  --smoke-max-n-epochs <int>
  --smoke-max-burnin <int>

Use "--" to pass any additional arguments directly to sweep_np.py/sweep_sp.py.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_simulations.sbatch"

PROFILE=""
CONFIG=""
ALL_CONFIGS=0
ALL_NP_CONFIGS=0
ALL_SP_CONFIGS=0
MODE="auto"
NODE_ID=0
N_NODES=1

RUNTIME_ARGS=()
EXTRA_SWEEP_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --all-configs)
      ALL_CONFIGS=1
      shift
      ;;
    --all-np-configs)
      ALL_NP_CONFIGS=1
      shift
      ;;
    --all-sp-configs)
      ALL_SP_CONFIGS=1
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
    --n-experiments|--seed|--smoke-n-experiments|--smoke-max-n-samples|--smoke-max-n-test|--smoke-max-n-epochs|--smoke-max-burnin)
      RUNTIME_ARGS+=("$1" "$2")
      shift 2
      ;;
    --smoke-test|--force-rerun)
      RUNTIME_ARGS+=("$1")
      shift
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

if [[ -z "${PROFILE}" ]]; then
  echo "--profile is required." >&2
  usage
  exit 1
fi

SELECTION_COUNT=0
if [[ -n "${CONFIG}" ]]; then
  ((SELECTION_COUNT += 1))
fi
if [[ ${ALL_CONFIGS} -eq 1 ]]; then
  ((SELECTION_COUNT += 1))
fi
if [[ ${ALL_NP_CONFIGS} -eq 1 ]]; then
  ((SELECTION_COUNT += 1))
fi
if [[ ${ALL_SP_CONFIGS} -eq 1 ]]; then
  ((SELECTION_COUNT += 1))
fi
if [[ ${SELECTION_COUNT} -ne 1 ]]; then
  echo "Provide exactly one of: --config, --all-configs, --all-np-configs, --all-sp-configs." >&2
  exit 1
fi

PARTITION=""
TIME_LIMIT=""
CPUS=""
MEM=""

case "${PROFILE}" in
  sapphire)
    PARTITION="sapphire"
    TIME_LIMIT="2-12:00"
    CPUS="110"
    MEM="128000"
    ;;
  test)
    PARTITION="test"
    TIME_LIMIT="0-10:30"
    CPUS="110"
    MEM="128000"
    ;;
  shared)
    PARTITION="shared"
    TIME_LIMIT="2-12:00"
    CPUS="48"
    MEM="100000"
    ;;
  unrestricted)
    PARTITION="unrestricted"
    TIME_LIMIT="30-10:30"
    CPUS="48"
    MEM="180000"
    ;;
  intermediate)
    PARTITION="intermediate"
    TIME_LIMIT="7-00:00"
    CPUS="110"
    MEM="180000"
    ;;
  *)
    echo "Invalid profile '${PROFILE}'. Expected: sapphire|test|shared|unrestricted|intermediate." >&2
    exit 1
    ;;
esac

SBATCH_CMD=(
  sbatch
  --chdir "${SCRIPT_DIR}"
  --partition "${PARTITION}"
  --time "${TIME_LIMIT}"
  --cpus-per-task "${CPUS}"
  --mem "${MEM}"
)

if [[ ${ALL_CONFIGS} -eq 1 || ${ALL_NP_CONFIGS} -eq 1 || ${ALL_SP_CONFIGS} -eq 1 ]]; then
  CONFIG_GLOB="config_*.py"
  if [[ ${ALL_NP_CONFIGS} -eq 1 ]]; then
    CONFIG_GLOB="config_np_*.py"
  elif [[ ${ALL_SP_CONFIGS} -eq 1 ]]; then
    CONFIG_GLOB="config_sp_*.py"
  fi

  CONFIG_FILES=()
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    CONFIG_FILES+=("${line}")
  done < <(
    cd "${SCRIPT_DIR}"
    for f in ${CONFIG_GLOB}; do
      [[ -e "${f}" ]] || continue
      echo "${f}"
    done | sort
  )
  if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
    echo "No config_*.py files found in ${SCRIPT_DIR}." >&2
    exit 1
  fi

  LIST_DIR="${SCRIPT_DIR}/.slurm_config_lists"
  mkdir -p "${LIST_DIR}"
  LIST_FILE="${LIST_DIR}/configs_$(date +%Y%m%d_%H%M%S)_$RANDOM.txt"

  : > "${LIST_FILE}"
  for cfg in "${CONFIG_FILES[@]}"; do
    printf '%s\n' "${cfg%.py}" >> "${LIST_FILE}"
  done

  ARRAY_RANGE="0-$(( ${#CONFIG_FILES[@]} - 1 ))"
  SBATCH_CMD+=(--array "${ARRAY_RANGE}")

  SBATCH_CMD+=("${SBATCH_SCRIPT}" --config-list "${LIST_FILE}" --mode "${MODE}" --node-id "${NODE_ID}" --n-nodes "${N_NODES}")
else
  SBATCH_CMD+=("${SBATCH_SCRIPT}" --config "${CONFIG}" --mode "${MODE}" --node-id "${NODE_ID}" --n-nodes "${N_NODES}")
fi

if [[ ${#RUNTIME_ARGS[@]} -gt 0 ]]; then
  SBATCH_CMD+=("${RUNTIME_ARGS[@]}")
fi
if [[ ${#EXTRA_SWEEP_ARGS[@]} -gt 0 ]]; then
  SBATCH_CMD+=(-- "${EXTRA_SWEEP_ARGS[@]}")
fi

printf 'Submitting command:'
printf ' %q' "${SBATCH_CMD[@]}"
printf '\n'

"${SBATCH_CMD[@]}"
