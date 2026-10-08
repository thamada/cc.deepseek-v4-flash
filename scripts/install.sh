#!/usr/bin/env bash
# Install the llama.cpp runtime (ROCm / CUDA / Metal / CPU) and a
# DeepSeek-V4-Flash-0731 GGUF that can run with 32 GiB of VRAM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

install_apt() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        return
    fi
    if command -v cmake >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
        return
    fi
    command -v apt-get >/dev/null 2>&1 || die "cmake / ninja / g++ が無く、apt-get も使えません"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y build-essential cmake ninja-build curl libcurl4-openssl-dev git python3 python3-venv python3-pip pkg-config ca-certificates
}

ensure_venv() {
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        python3 -m venv "${VENV_DIR}"
    fi
    "${VENV_DIR}/bin/pip" install -q -U pip huggingface_hub
}

clone_llama() {
    if [[ -f "${SRC_DIR}/CMakeLists.txt" ]]; then
        return
    fi
    mkdir -p "${ROOT}/third_party"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${SRC_DIR}"
}

cmake_configure() {
    local device="$1"
    local target="$2"
    local -a args=(
        -S "${SRC_DIR}"
        -B "${BUILD_DIR}"
        -G Ninja
        -DCMAKE_BUILD_TYPE=Release
        -DLLAMA_CURL=ON
    )
    case "${device}" in
        cpu)
            args+=(-DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_METAL=OFF)
            ;;
        cuda)
            command -v nvcc >/dev/null 2>&1 || die "DEEPSEEK_DEVICE=cuda ですが nvcc が見つかりません"
            args+=(-DGGML_CUDA=ON)
            ;;
        mps)
            [[ "$(uname -s)" == "Darwin" ]] || die "DEEPSEEK_DEVICE=mps は macOS 専用です"
            args+=(-DGGML_METAL=ON)
            ;;
        rocm|rocm-gfx120|rocm-gfx1151|rocm-gfx110)
            command -v hipconfig >/dev/null 2>&1 || die "ROCm の hipconfig が見つかりません"
            local hip_path hip_bin
            hip_path="$(hipconfig -R)"
            hip_bin="$(hipconfig -l)"
            export HIPCXX="${hip_bin}/clang"
            export HIP_PATH="${hip_path}"
            export HIP_DEVICE_LIB_PATH="${hip_path}/lib/llvm/amdgcn/bitcode"
            [[ -x "${HIPCXX}" ]] || die "HIP コンパイラがありません: ${HIPCXX}"
            args+=(-DGGML_HIP=ON -DGPU_TARGETS="${target}")
            ;;
    esac
    cmake "${args[@]}"
}

build_llama() {
    local device="$1"
    local target="$2"
    mkdir -p "${STAMP_DIR}"
    local rev stamp
    rev="$(git -C "${SRC_DIR}" rev-parse HEAD)"
    stamp="${device}|${target}|${rev}"
    if [[ -x "$(llama_cli)" && -x "$(llama_server)" && -f "${STAMP_DIR}/build.stamp" ]] &&
        [[ "$(cat "${STAMP_DIR}/build.stamp")" == "${stamp}" ]]; then
        echo "推論エンジンは構築済みです (${device} ${target})"
        return
    fi
    echo "llama.cpp をビルドします: device=${device} target=${target:-cpu}"
    cmake_configure "${device}" "${target}"
    local jobs
    jobs="$(nproc)"
    # Keep a little RAM free on 16 GiB machines while HIP objects compile.
    if (( jobs > 6 )); then
        jobs=6
    fi
    cmake --build "${BUILD_DIR}" --config Release -j "${jobs}" --target llama-cli llama-server
    echo "${stamp}" > "${STAMP_DIR}/build.stamp"
}

download_model() {
    local quant="$1"
    mkdir -p "${STAMP_DIR}" "${MODEL_DIR}"
    if model_ready "${quant}"; then
        echo "モデルは取得済みです (${quant})"
        echo "${quant}" > "${STAMP_DIR}/model.stamp"
        return
    fi
    echo "モデルを取得します: ${MODEL_REPO} / ${quant}"
    echo "VRAM ${DEEPSEEK_VRAM_GIB} GiB では MoE 全体は載らないため、エキスパートは CPU に置きます。"
    ensure_venv
    "${VENV_DIR}/bin/python" - "${MODEL_REPO}" "${MODEL_DIR}" "${quant}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, dest, quant = sys.argv[1:]
snapshot_download(
    repo_id=repo,
    local_dir=dest,
    allow_patterns=[f"{quant}/*"],
)
PY
    model_ready "${quant}" || die "モデルの取得後もシャードが揃っていません: ${quant}"
    echo "${quant}" > "${STAMP_DIR}/model.stamp"
}

main() {
    install_apt
    local device target quant
    device="$(detect_device)"
    target="$(gpu_target_for_device "${device}")"
    quant="$(select_quant)"
    local ram
    ram="$(mem_total_gib)"
    echo "device=${device} gpu_target=${target:-none} quant=${quant} ram=${ram}GiB vram=${DEEPSEEK_VRAM_GIB}GiB"
    clone_llama
    # Weight download dominates wall time. Compile while it runs.
    download_model "${quant}" &
    local dpid=$!
    local build_rc=0
    build_llama "${device}" "${target}" || build_rc=$?
    wait "${dpid}" || {
        local dl_rc=$?
        [[ "${build_rc}" -eq 0 ]] || exit "${build_rc}"
        exit "${dl_rc}"
    }
    [[ "${build_rc}" -eq 0 ]] || exit "${build_rc}"
    echo "install 完了: $(model_gguf "${quant}")"
}

main "$@"
