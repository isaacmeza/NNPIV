#!/bin/bash
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
#SBATCH -p sapphire
#SBATCH -t 0-10:00
#SBATCH -c 110
#SBATCH --mem=900000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=i.mezal.12@gmail.com

set -euo pipefail

format_duration() {
  local total_seconds="$1"
  local hours=$(( total_seconds / 3600 ))
  local minutes=$(( (total_seconds % 3600) / 60 ))
  local seconds=$(( total_seconds % 60 ))
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

usage() {
  cat <<'USAGE'
Usage:
  run_simulations.sbatch --config <config_name_or_path> [options]
  run_simulations.sbatch --config-list <path> [options]   # for Slurm array mode

Options:
  --mode <auto|np|sp>         Runner mode (default: auto by filename prefix)
  --node-id <int>             Placeholder replacement for __NODEID__ (default: 0)
  --n-nodes <int>             Placeholder replacement for __NNODES__ (default: 1)

Runtime overrides (forwarded to sweep_np.py/sweep_sp.py):
  --n-experiments <int>
  --seed <int>
  --smoke-test
  --force-rerun
  --smoke-n-experiments <int>  (default: 5)
  --smoke-max-n-samples <int>
  --smoke-max-n-test <int>
  --smoke-max-n-epochs <int>   (default: 150)
  --smoke-max-burnin <int>     (default: 100)

Use "--" to pass any additional arguments directly to the sweep driver.
USAGE
}

SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_SOURCE_DIR}"
if ! compgen -G "${SCRIPT_DIR}/config_*.py" > /dev/null; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && compgen -G "${SLURM_SUBMIT_DIR}/config_*.py" > /dev/null; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  elif compgen -G "$(pwd)/config_*.py" > /dev/null; then
    SCRIPT_DIR="$(pwd)"
  fi
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="auto"
CONFIG=""
CONFIG_LIST=""
NODE_ID=0
N_NODES=1

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

trim_config_token() {
  local value="$1"
  # Normalize accidental CRLF/whitespace from copied commands or list files.
  value="${value//$'\r'/}"
  value="${value//$'\n'/}"
  # Trim leading/trailing whitespace.
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf "%s" "${value}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --config-list)
      CONFIG_LIST="$2"
      shift 2
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

if [[ -n "${CONFIG}" && -n "${CONFIG_LIST}" ]]; then
  echo "Provide exactly one of --config or --config-list." >&2
  exit 1
fi
if [[ -z "${CONFIG}" && -z "${CONFIG_LIST}" ]]; then
  echo "You must provide one of --config or --config-list." >&2
  exit 1
fi

if [[ -n "${CONFIG_LIST}" ]]; then
  if [[ ! -f "${CONFIG_LIST}" ]]; then
    echo "Config list file not found: ${CONFIG_LIST}" >&2
    exit 1
  fi
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "--config-list requires Slurm array mode (SLURM_ARRAY_TASK_ID missing)." >&2
    exit 1
  fi

  CONFIGS=()
  while IFS= read -r line; do
    line="$(trim_config_token "${line}")"
    [[ -n "${line}" ]] || continue
    CONFIGS+=("${line}")
  done < "${CONFIG_LIST}"
  if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "Config list file is empty: ${CONFIG_LIST}" >&2
    exit 1
  fi

  TASK_ID="${SLURM_ARRAY_TASK_ID}"
  if (( TASK_ID < 0 || TASK_ID >= ${#CONFIGS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0, $(( ${#CONFIGS[@]} - 1 ))]." >&2
    exit 1
  fi

  CONFIG="${CONFIGS[${TASK_ID}]}"
fi

CONFIG="$(trim_config_token "${CONFIG}")"

resolve_config_path() {
  local raw
  raw="$(trim_config_token "$1")"
  local trimmed="${raw%.py}"
  trimmed="${trimmed#./}"
  trimmed="$(trim_config_token "${trimmed}")"

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

  if [[ "${trimmed}" == config_* ]]; then
    candidates+=("${SCRIPT_DIR}/${trimmed}.py")
  fi

  local c
  for c in "${candidates[@]}"; do
    if [[ -f "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done

  # Final fallback for absolute/relative paths already including extension.
  if [[ -f "${raw}" ]]; then
    echo "${raw}"
    return 0
  fi

  return 1
}

CONFIG_PATH="$(resolve_config_path "${CONFIG}")" || {
  echo "Could not resolve config from input: ${CONFIG}" >&2
  exit 1
}

CONFIG_BASENAME="$(basename "${CONFIG_PATH}" .py)"

if [[ "${MODE}" == "auto" ]]; then
  case "${CONFIG_BASENAME}" in
    config_np_*)
      MODE="np"
      ;;
    config_sp_*)
      MODE="sp"
      ;;
    *)
      echo "Unable to infer mode for ${CONFIG_BASENAME}. Use --mode np|sp." >&2
      exit 1
      ;;
  esac
fi

if [[ "${MODE}" != "np" && "${MODE}" != "sp" ]]; then
  echo "Invalid --mode ${MODE}; expected auto|np|sp." >&2
  exit 1
fi

# Avoid oversubscription: cap BLAS/OpenMP threads to 1 per process
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CLUSTER_PYTHON_MODULE="${CLUSTER_PYTHON_MODULE:-python/3.13}"
module load "${CLUSTER_PYTHON_MODULE}"
mamba activate nnpiv_jasa

cd "${SCRIPT_DIR}"
TMP_DIR="temp"
TMP_FILE="${NODE_ID}_${CONFIG_BASENAME}"
TMP_PATH="${TMP_DIR}/${TMP_FILE}.py"
mkdir -p "${TMP_DIR}"
cp "${CONFIG_PATH}" "${TMP_PATH}"

cleanup() {
  rm -f "${TMP_PATH}" "${TMP_PATH}.bak"
}
trap cleanup EXIT

sed -i.bak "s/__NODEID__/${NODE_ID}/g" "${TMP_PATH}"
sed -i.bak "s/__NNODES__/${N_NODES}/g" "${TMP_PATH}"
rm -f "${TMP_PATH}.bak"

if [[ "${MODE}" == "np" ]]; then
  RUNNER="sweep_np.py"
else
  RUNNER="sweep_sp.py"
fi

RUNNER_ARGS=(--config "temp.${TMP_FILE}")

if [[ -n "${N_EXPERIMENTS}" ]]; then
  RUNNER_ARGS+=(--n-experiments "${N_EXPERIMENTS}")
fi
if [[ -n "${SEED}" ]]; then
  RUNNER_ARGS+=(--seed "${SEED}")
fi
if [[ "${SMOKE_TEST}" -eq 1 ]]; then
  RUNNER_ARGS+=(
    --smoke-test
    --smoke-n-experiments "${SMOKE_N_EXPERIMENTS}"
    --smoke-max-n-samples "${SMOKE_MAX_N_SAMPLES}"
    --smoke-max-n-test "${SMOKE_MAX_N_TEST}"
    --smoke-max-n-epochs "${SMOKE_MAX_N_EPOCHS}"
    --smoke-max-burnin "${SMOKE_MAX_BURNIN}"
  )
fi
if [[ "${FORCE_RERUN}" -eq 1 ]]; then
  RUNNER_ARGS+=(--force-rerun)
fi

if [[ ${#EXTRA_SWEEP_ARGS[@]} -gt 0 ]]; then
  RUNNER_ARGS+=("${EXTRA_SWEEP_ARGS[@]}")
fi

START_TS="$(date +%s)"
echo "Running ${CONFIG_BASENAME} (mode=${MODE})"

if srun python "${RUNNER}" "${RUNNER_ARGS[@]}"; then
  STATUS_CODE=0
  STATUS_TEXT="ok"
else
  STATUS_CODE=$?
  STATUS_TEXT="failed(${STATUS_CODE})"
fi

END_TS="$(date +%s)"
ELAPSED_SECONDS=$(( END_TS - START_TS ))
echo "Finished ${CONFIG_BASENAME} in $(format_duration "${ELAPSED_SECONDS}") (${ELAPSED_SECONDS}s) [${STATUS_TEXT}]"

exit "${STATUS_CODE}"
