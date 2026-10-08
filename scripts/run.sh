#!/usr/bin/env bash
# Run DeepSeek-V4-Flash.
#   make run                         one-shot generation
#   make run ARGS="--listen 0.0.0.0" OpenAI-compatible server
#
# Environment:
#   DEEPSEEK_DEVICE=cpu|cuda|rocm|rocm-gfx120|rocm-gfx1151|rocm-gfx110|mps
#   DEEPSEEK_LISTEN=127.0.0.1
#   DEEPSEEK_PORT=8188          (DEEPSEEKK_PORT is accepted as an alias)
#   DEEPSEEK_MANAGER=0          1 backgrounds the server
#   DEEPSEEK_PROMPT=...
#   DEEPSEEK_CTX=8192
#   DEEPSEEK_N=128
#   DEEPSEEK_MOE_CACHE_MIB=8192 GPU cache for CPU-resident experts
#   DEEPSEEK_REASONING=off|on|auto
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

HOST="${DEEPSEEK_LISTEN:-127.0.0.1}"
PORT="${DEEPSEEK_PORT:-${DEEPSEEKK_PORT:-8188}}"
MODE="cli"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --listen)
            MODE="server"
            if [[ $# -ge 2 && "$2" != --* ]]; then
                HOST="$2"
                shift 2
            else
                shift
            fi
            ;;
        --listen=*)
            MODE="server"
            HOST="${1#--listen=}"
            shift
            ;;
        *)
            EXTRA+=("$1")
            shift
            ;;
    esac
done

device="$(detect_device)"
apply_hsa_override "${device}" || true
quant="$(select_quant)"
model="$(model_gguf "${quant}")"

case "${device}" in
    rocm|rocm-gfx120|rocm-gfx1151|rocm-gfx110)
        if command -v hipconfig >/dev/null 2>&1; then
            export HIP_PATH="$(hipconfig -R)"
            export ROCM_PATH="${HIP_PATH}"
        fi
        rocm_lib="/opt/rocm/lib"
        if [[ -d "${HIP_PATH:-}/lib" ]]; then
            rocm_lib="${HIP_PATH}/lib:${rocm_lib}"
        fi
        export LD_LIBRARY_PATH="${rocm_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export PATH="/opt/rocm/bin:${PATH}"
        # HIP graphs on this ROCm stack can sit forever inside the HSA runtime
        # after the weights are already on the device.
        export GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}"
        ;;
esac

CTX="${DEEPSEEK_CTX:-4096}"
if [[ -n "${DEEPSEEK_MOE_CACHE_MIB:-}" ]]; then
    MOE_CACHE="${DEEPSEEK_MOE_CACHE_MIB}"
else
    # An expert cache needs a large HSA allocation. On a 32 GiB GPU with
    # little system RAM that call can sit inside the ROCm runtime, so keep
    # the cache off unless the host has room for the experts' page cache.
    if (( "$(mem_total_gib)" >= 64 )); then
        MOE_CACHE=8192
    else
        MOE_CACHE=0
    fi
fi
THREADS="${DEEPSEEK_THREADS:-$(nproc)}"
# 32 GiB card: attention, shared expert, embeddings and KV stay on the GPU.
# Routed experts stay in the mmap'd GGUF on the host. A slice of VRAM caches
# the experts that were just used.
COMMON=(
    -m "${model}"
    --jinja
    --temp 1.0
    --top-p 1.0
    --min-p 0.01
    -c "${CTX}"
    -ngl all
    --cpu-moe
    --fit off
    --threads "${THREADS}"
    --moe-cache-mib "${MOE_CACHE}"
)

run_logged() {
    local log="$1"
    shift
    mkdir -p "${ROOT}/var"
    local -a cmd=("$@")
    if command -v stdbuf >/dev/null 2>&1; then
        cmd=(stdbuf -o0 -eL "$@")
    fi
    set +e
    "${cmd[@]}" 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    set -e
    return "${rc}"
}

if [[ "${MODE}" == "server" ]]; then
    bin="$(llama_server)"
    [[ -x "${bin}" ]] || die "llama-server がありません。make install を先に実行してください"
    args=(
        "${bin}"
        "${COMMON[@]}"
        --host "${HOST}"
        --port "${PORT}"
        --reasoning "${DEEPSEEK_REASONING:-auto}"
        "${EXTRA[@]}"
    )
    echo "サーバを起動します: http://${HOST}:${PORT}/"
    if [[ "${DEEPSEEK_MANAGER:-0}" == "1" ]]; then
        mkdir -p "${ROOT}/var"
        if [[ -f "${ROOT}/var/server.pid" ]] && kill -0 "$(cat "${ROOT}/var/server.pid")" 2>/dev/null; then
            die "サーバは既に動いています (pid $(cat "${ROOT}/var/server.pid"))"
        fi
        nohup "${args[@]}" > "${ROOT}/var/server.log" 2>&1 &
        echo $! > "${ROOT}/var/server.pid"
        echo "バックグラウンド起動 pid=$(cat "${ROOT}/var/server.pid") ログ=${ROOT}/var/server.log"
        exit 0
    fi
    exec "${args[@]}"
fi

bin="$(llama_cli)"
[[ -x "${bin}" ]] || die "llama-cli がありません。make install を先に実行してください"
PROMPT="${DEEPSEEK_PROMPT:-こんにちは。自己紹介を一文でしてください。}"
NPREDICT="${DEEPSEEK_N:-32}"
args=(
    "${bin}"
    "${COMMON[@]}"
    --single-turn
    --reasoning "${DEEPSEEK_REASONING:-off}"
    -n "${NPREDICT}"
    -p "${PROMPT}"
    "${EXTRA[@]}"
)

echo "推論を実行します: quant=${quant} ctx=${CTX} moe_cache=${MOE_CACHE}MiB"
log="${ROOT}/var/run.log"
if run_logged "${log}" "${args[@]}"; then
    exit 0
fi
if grep -q -E 'flash_attn|Flash Attention|out of memory|hipErrorOutOfMemory|HIP out of memory' "${log}"; then
    echo "GPU 側の起動に失敗したため、Flash Attention と MoE キャッシュを外して再試行します" >&2
    fallback=()
    for a in "${args[@]}"; do
        if [[ "${a}" == "--moe-cache-mib" ]]; then
            skip_next=1
            continue
        fi
        if [[ "${skip_next:-0}" == "1" ]]; then
            skip_next=0
            continue
        fi
        fallback+=("${a}")
    done
    fallback+=(--flash-attn off --moe-cache-mib 0)
    run_logged "${log}" "${fallback[@]}"
    exit $?
fi
exit 1
