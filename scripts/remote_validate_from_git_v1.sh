#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/remote_validate_from_git_v1.sh [options] -- <command> [args...]

Clone the current git branch on the data processing machine and run a command
from that fresh clone. This is for validating pushed code against shared
storage/data-machine dependencies.

Options:
  --remote-host HOST        SSH host for data processing machine (default: guangzhoujump)
  --remote-root PATH        Parent dir for remote validation clones
                            (default: /mnt/kai_kpfs/weilai/train/remote-validations)
  --repo-url URL            Git URL to clone (default: local origin remote)
  --branch BRANCH          Branch to clone (default: current branch)
  --venv PATH              Venv to source before command
                            (default: /mnt/kai_kpfs/weilai/train/pt-data-pipeline/.venv)
  --run-id ID              Remote run dir suffix (default: branch + timestamp)
  --cleanup                Remove the remote clone after the command exits
  -h, --help               Show this help

Example:
  bash scripts/remote_validate_from_git_v1.sh -- \
    python3 scripts/validate_source_registry_v1.py --config-dir configs/sources --check-paths
EOF
}

REMOTE_HOST="guangzhoujump"
REMOTE_ROOT="/mnt/kai_kpfs/weilai/train/remote-validations"
REPO_URL="$(git config --get remote.origin.url || true)"
BRANCH="$(git branch --show-current || true)"
VENV="/mnt/kai_kpfs/weilai/train/pt-data-pipeline/.venv"
RUN_ID=""
CLEANUP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --remote-root)
      REMOTE_ROOT="$2"
      shift 2
      ;;
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --venv)
      VENV="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --cleanup)
      CLEANUP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${REPO_URL}" ]]; then
  echo "could not infer repo URL; pass --repo-url" >&2
  exit 2
fi

if [[ -z "${BRANCH}" ]]; then
  echo "could not infer current branch; pass --branch" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  echo "missing remote command after --" >&2
  usage >&2
  exit 2
fi

if [[ -z "${RUN_ID}" ]]; then
  SAFE_BRANCH="${BRANCH//\//_}"
  RUN_ID="${SAFE_BRANCH}_$(date +%Y%m%d_%H%M%S)"
fi

echo "remote_host=${REMOTE_HOST}"
echo "repo_url=${REPO_URL}"
echo "branch=${BRANCH}"
echo "remote_root=${REMOTE_ROOT}"
echo "run_id=${RUN_ID}"
echo

ssh "${REMOTE_HOST}" \
  REPO_URL="${REPO_URL}" \
  BRANCH="${BRANCH}" \
  REMOTE_ROOT="${REMOTE_ROOT}" \
  RUN_ID="${RUN_ID}" \
  VENV="${VENV}" \
  CLEANUP="${CLEANUP}" \
  bash -s -- "$@" <<'REMOTE_SCRIPT'
set -euo pipefail

RUN_DIR="${REMOTE_ROOT}/${RUN_ID}"
REPO_DIR="${RUN_DIR}/repo"

if [[ -e "${RUN_DIR}" ]]; then
  echo "remote run dir already exists: ${RUN_DIR}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
echo "cloning ${REPO_URL} branch ${BRANCH}"
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"

cd "${REPO_DIR}"
echo "remote clone: ${REPO_DIR}"
git rev-parse --short HEAD

if [[ -n "${VENV}" && -f "${VENV}/bin/activate" ]]; then
  # The fresh clone intentionally reuses the prepared data-machine venv.
  source "${VENV}/bin/activate"
else
  echo "warning: venv not found, running with host python: ${VENV}" >&2
fi

echo "+ $*"
set +e
"$@"
STATUS=$?
set -e

if [[ "${CLEANUP}" == "1" ]]; then
  cd /
  rm -rf "${RUN_DIR}"
  echo "cleaned ${RUN_DIR}"
else
  echo "kept remote clone at ${REPO_DIR}"
fi

exit "${STATUS}"
REMOTE_SCRIPT
