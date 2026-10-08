#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GROUNDINGPI_ROOT="${GROUNDINGPI_ROOT:-/home/admin1/GroundingPI/vla/starvla}"

python -m pip install -e "${XPL_ROOT}"
# The policy process only imports GroundingPI's websocket client.  Qwen3,
# torch and flash-attn belong to the separately managed GroundingPI server env.
python -m pip install "websockets>=14" "msgpack>=1" "msgpack-numpy>=0.4.8" "Pillow>=9"

if [[ ! -f "${GROUNDINGPI_ROOT}/deployment/model_server/tools/websocket_policy_client.py" ]]; then
    echo "[GroundingPI_QwenPI] warning: websocket client not found under ${GROUNDINGPI_ROOT}" >&2
    echo "[GroundingPI_QwenPI] set groundingpi_root in deploy.yml to GroundingPI/vla/starvla" >&2
fi
