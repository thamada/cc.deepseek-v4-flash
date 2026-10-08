#!/usr/bin/env bash
# Remove the llama.cpp build. The downloaded GGUF is kept unless
# DEEPSEEK_CLEAN_MODELS=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ -f "${ROOT}/var/server.pid" ]]; then
    pid="$(cat "${ROOT}/var/server.pid")"
    if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" || true
    fi
fi

rm -rf "${BUILD_DIR}" "${STAMP_DIR}" "${ROOT}/var"
if [[ "${DEEPSEEK_CLEAN_MODELS:-0}" == "1" ]]; then
    rm -rf "${MODEL_DIR}" "${VENV_DIR}"
fi
echo "clean 完了"
