# Shared paths and device selection for DeepSeek-V4-Flash.
# shellcheck shell=bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT}/third_party/llama.cpp"
BUILD_DIR="${SRC_DIR}/build"
BIN_DIR="${BUILD_DIR}/bin"
MODEL_REPO="unsloth/DeepSeek-V4-Flash-0731-GGUF"
MODEL_DIR="${ROOT}/models/DeepSeek-V4-Flash-0731-GGUF"
STAMP_DIR="${ROOT}/.install"
VENV_DIR="${ROOT}/.venv"

# The card is treated as 32 GiB even if the driver reports a slightly
# different figure (R9700 shows about 32624 MiB).
DEEPSEEK_VRAM_GIB="${DEEPSEEK_VRAM_GIB:-32}"

die() {
    echo "error: $*" >&2
    exit 1
}

detect_gfx() {
    if ! command -v rocminfo >/dev/null 2>&1; then
        return 0
    fi
    rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx/{print $2; exit}'
}

detect_device() {
    if [[ -n "${DEEPSEEK_DEVICE:-}" ]]; then
        echo "${DEEPSEEK_DEVICE}"
        return
    fi
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "mps"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "cuda"
        return
    fi
    local gfx
    gfx="$(detect_gfx || true)"
    case "${gfx}" in
        gfx120*) echo "rocm-gfx120" ;;
        gfx1151|gfx1152|gfx1153) echo "rocm-gfx1151" ;;
        gfx110*) echo "rocm-gfx110" ;;
        gfx*) echo "rocm" ;;
        *) echo "cpu" ;;
    esac
}

# ISA passed to llama.cpp -DGPU_TARGETS.
gpu_target_for_device() {
    local device="$1"
    local gfx
    gfx="$(detect_gfx || true)"
    case "${device}" in
        cpu|cuda|mps)
            echo ""
            ;;
        rocm-gfx120)
            if [[ "${gfx}" == gfx120* ]]; then
                echo "${gfx}"
            else
                echo "gfx1201"
            fi
            ;;
        rocm-gfx1151)
            echo "gfx1151"
            ;;
        rocm-gfx110)
            if [[ "${gfx}" == gfx110* ]]; then
                echo "${gfx}"
            else
                echo "gfx1100"
            fi
            ;;
        rocm)
            [[ -n "${gfx}" ]] || die "DEEPSEEK_DEVICE=rocm ですが rocminfo から GPU を検出できません。GPU_ARCH を持つ ROCm 環境が必要です"
            echo "${gfx}"
            ;;
        *)
            die "未知の DEEPSEEK_DEVICE=${device}（cpu|cuda|rocm|rocm-gfx120|rocm-gfx1151|rocm-gfx110|mps）"
            ;;
    esac
}

# gfx1152/gfx1153 have no rocBLAS Tensile library; run the gfx1151 image.
apply_hsa_override() {
    local device="$1"
    local gfx
    gfx="$(detect_gfx || true)"
    if [[ "${device}" == "rocm-gfx1151" && ( "${gfx}" == "gfx1152" || "${gfx}" == "gfx1153" ) ]]; then
        export HSA_OVERRIDE_GFX_VERSION="11.5.1"
        echo "HSA_OVERRIDE_GFX_VERSION=11.5.1 （${gfx} を gfx1151 として実行）"
    fi
}

mem_total_gib() {
    awk '/MemTotal:/ {printf "%d", $2/1024/1024}' /proc/meminfo
}

# Smallest Unsloth quant whose file size fits in RAM + the 32 GiB GPU,
# with a few GiB of headroom for KV and compute buffers.
# Experts are mmap'd on the CPU, so this is a comfort threshold rather
# than a hard VRAM limit. Override with DEEPSEEK_QUANT.
select_quant() {
    if [[ -n "${DEEPSEEK_QUANT:-}" ]]; then
        echo "${DEEPSEEK_QUANT}"
        return
    fi
    local ram vram total
    ram="$(mem_total_gib)"
    vram="${DEEPSEEK_VRAM_GIB}"
    total=$((ram + vram))
    if (( total >= 175 )); then
        echo "UD-Q8_K_XL"
    elif (( total >= 165 )); then
        echo "UD-Q4_K_XL"
    elif (( total >= 120 )); then
        echo "UD-IQ3_XXS"
    elif (( total >= 100 )); then
        echo "UD-IQ2_XXS"
    else
        echo "UD-IQ1_S"
    fi
}

model_gguf() {
    local quant="$1"
    local dir="${MODEL_DIR}/${quant}"
    local first
    first="$(find "${dir}" -maxdepth 1 -name '*-00001-of-*.gguf' -print -quit 2>/dev/null || true)"
    [[ -n "${first}" ]] || die "モデルが見つかりません: ${dir}"
    echo "${first}"
}

model_ready() {
    local quant="$1"
    local dir="${MODEL_DIR}/${quant}"
    local first base n i shard
    first="$(find "${dir}" -maxdepth 1 -name '*-00001-of-*.gguf' -print -quit 2>/dev/null || true)"
    [[ -n "${first}" ]] || return 1
    base="${first%-00001-of-*.gguf}"
    n="${first##*-of-}"
    n="${n%.gguf}"
    n=$((10#${n}))
    for ((i = 1; i <= n; i++)); do
        printf -v shard "%s-%05d-of-%05d.gguf" "${base}" "${i}" "${n}"
        [[ -s "${shard}" ]] || return 1
    done
    # The first split is metadata-sized. A later shard must be a real weight file.
    [[ "${n}" -ge 2 ]] || return 0
    printf -v shard "%s-%05d-of-%05d.gguf" "${base}" 2 "${n}"
    local bytes
    bytes="$(stat -c %s "${shard}")"
    if (( bytes <= 1000000000 )); then
        return 1
    fi
    return 0
}

llama_cli() { echo "${BIN_DIR}/llama-cli"; }
llama_server() { echo "${BIN_DIR}/llama-server"; }
