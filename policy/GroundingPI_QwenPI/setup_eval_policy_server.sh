#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 9 || $# -gt 10 ]]; then
  echo "Usage: bash $0 <bench> <task> <ckpt> <env_cfg> <action_type> <seed> <policy_gpu> <policy_env_or_python> <port> [host]" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
policy_env=$8
if [[ -x "${policy_env}/bin/python" ]]; then
  POLICY_PYTHON="${policy_env}/bin/python"
elif [[ -f "${policy_env}" && -x "${policy_env}" ]]; then
  POLICY_PYTHON="${policy_env}"
elif [[ "${policy_env}" == "uv" && -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  POLICY_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${policy_env}"
  POLICY_PYTHON="$(command -v python)"
fi
export PYTHONPATH="${XPL_ROOT}:${PYTHONPATH:-}"

# Read startup values from deploy.yml, then apply explicit environment overrides.
settings="$("${POLICY_PYTHON}" - "${SCRIPT_DIR}" "$1" "$3" "$4" "$5" "$6" <<'PY'
import os, sys
from pathlib import Path
import yaml
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
folder = Path(sys.argv[1])
cfg = yaml.safe_load((folder / 'deploy.yml').read_text())
cfg.update(zip(('bench_name', 'ckpt_name', 'env_cfg_type', 'action_type', 'seed'), sys.argv[2:]))
def value(env, key, default):
    return os.environ.get(env, cfg.get(key) if cfg.get(key) is not None else default)
def path(value):
    p = Path(str(value)).expanduser()
    return str(p if p.is_absolute() else (folder / p).resolve())
autostart = str(value('GROUNDINGPI_SERVER_AUTOSTART', 'groundingpi_server_autostart', False)).lower()
checkpoint = value('GROUNDINGPI_CHECKPOINT_PATH', 'groundingpi_checkpoint_path', '')
if autostart in ('1', 'true'):
    if checkpoint:
        cfg['checkpoint_path'] = checkpoint
    checkpoint = str(resolve_checkpoint_root(cfg, folder / 'checkpoints'))
print(path(value('GROUNDINGPI_ROOT', 'groundingpi_root', '/home/admin1/GroundingPI/vla/starvla')))
print(value('GROUNDINGPI_SERVER_HOST', 'groundingpi_server_host', '127.0.0.1'))
print(value('GROUNDINGPI_SERVER_PORT', 'groundingpi_server_port', 5694))
print(autostart)
print(checkpoint)
print(value('GROUNDINGPI_PYTHON', 'groundingpi_python', sys.executable))
PY
)"
mapfile -t runtime <<< "${settings}"
GROUNDINGPI_ROOT=${runtime[0]}
GROUNDINGPI_SERVER_HOST=${runtime[1]}
GROUNDINGPI_SERVER_PORT=${runtime[2]}
GROUNDINGPI_SERVER_AUTOSTART=${runtime[3]}
GROUNDINGPI_CHECKPOINT_PATH=${runtime[4]}
GROUNDINGPI_PYTHON=${runtime[5]}

GROUNDINGPI_PID=""
POLICY_PID=""
cleanup() {
  for pid in "${POLICY_PID}" "${GROUNDINGPI_PID}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ "${GROUNDINGPI_SERVER_AUTOSTART}" == "1" || "${GROUNDINGPI_SERVER_AUTOSTART}" == "true" ]]; then
  [[ -f "${GROUNDINGPI_ROOT}/deployment/model_server/tools/websocket_policy_client.py" ]] || {
    echo "[SERVER][ERROR] invalid GroundingPI root: ${GROUNDINGPI_ROOT}" >&2; exit 1;
  }
  [[ -e "${GROUNDINGPI_CHECKPOINT_PATH}" ]] || {
    echo "[SERVER][ERROR] QwenPI checkpoint does not exist: ${GROUNDINGPI_CHECKPOINT_PATH}" >&2; exit 1;
  }
  echo "[SERVER] starting GroundingPI bridge on ${GROUNDINGPI_SERVER_HOST}:${GROUNDINGPI_SERVER_PORT}"
  (
    cd "${GROUNDINGPI_ROOT}"
    exec env CUDA_VISIBLE_DEVICES="$7" "${GROUNDINGPI_PYTHON}" "${SCRIPT_DIR}/groundingpi_server.py" \
      --groundingpi-root "${GROUNDINGPI_ROOT}" --ckpt-path "${GROUNDINGPI_CHECKPOINT_PATH}" \
      --host "${GROUNDINGPI_SERVER_HOST}" --port "${GROUNDINGPI_SERVER_PORT}" --use-bf16
  ) &
  GROUNDINGPI_PID=$!
  # Check readiness with the XPL interpreter, independent of the model environment.
  "${POLICY_PYTHON}" - "${GROUNDINGPI_SERVER_HOST}" "${GROUNDINGPI_SERVER_PORT}" "${GROUNDINGPI_PID}" <<'PY'
import os, socket, sys, time
host, port, pid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
deadline = time.monotonic() + float(os.environ.get('GROUNDINGPI_STARTUP_TIMEOUT', '1200'))
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit('GroundingPI bridge exited during startup')
    try:
        with socket.create_connection((host, port), timeout=0.5):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('Timed out waiting for GroundingPI bridge')
PY
fi

OVERRIDES=(
  "port=$9" "host=${10:-localhost}" "bench_name=$1" "task_name=$2" "ckpt_name=$3"
  "env_cfg_type=$4" "action_type=$5" "seed=$6" "gpu_id=$7" "policy_name=GroundingPI_QwenPI"
  "groundingpi_root=${GROUNDINGPI_ROOT}" "groundingpi_server_host=${GROUNDINGPI_SERVER_HOST}"
  "groundingpi_server_port=${GROUNDINGPI_SERVER_PORT}"
)
[[ -v GROUNDINGPI_INCLUDE_STATE ]] && OVERRIDES+=("include_state=${GROUNDINGPI_INCLUDE_STATE}")
[[ -v GROUNDINGPI_ALLOW_DUMMY_POLICY ]] && OVERRIDES+=("allow_dummy_policy=${GROUNDINGPI_ALLOW_DUMMY_POLICY}")
[[ -n "${GROUNDINGPI_DATASET_STATS_PATH:-}" ]] && OVERRIDES+=("dataset_statistics_path=${GROUNDINGPI_DATASET_STATS_PATH}")
[[ -n "${GROUNDINGPI_UNNORM_KEY:-}" ]] && OVERRIDES+=("unnorm_key=${GROUNDINGPI_UNNORM_KEY}")

CUDA_VISIBLE_DEVICES="$7" "${POLICY_PYTHON}" "${XPL_ROOT}/setup_policy_server.py" \
  --config_path "${SCRIPT_DIR}/deploy.yml" --overrides "${OVERRIDES[@]}" &
POLICY_PID=$!
wait "${POLICY_PID}"
