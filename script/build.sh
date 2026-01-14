#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${ROOT_DIR}/build"

CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"

# FastStencil
cmake -S "${ROOT_DIR}/2dstencil" -B "${BUILD_DIR}/FastStencil/2dstencil" "$CMAKE_FLAGS"
cmake --build "${BUILD_DIR}/FastStencil/2dstencil" -j
cmake -S "${ROOT_DIR}/3dstencil" -B "${BUILD_DIR}/FastStencil/3dstencil" "$CMAKE_FLAGS"
cmake --build "${BUILD_DIR}/FastStencil/3dstencil" -j
# EBISU
cmake -S "${ROOT_DIR}/third_party/EBISU/2dstencil" -B "${BUILD_DIR}/EBISU/2dstencil" "$CMAKE_FLAGS"
cmake --build "${BUILD_DIR}/EBISU/2dstencil" -j
cmake -S "${ROOT_DIR}/third_party/EBISU/3dstencil" -B "${BUILD_DIR}/EBISU/3dstencil" "$CMAKE_FLAGS"
cmake --build "${BUILD_DIR}/EBISU/3dstencil" -j
# ConvStencil
cmake -S "${ROOT_DIR}/third_party/ConvStencil" -B "${BUILD_DIR}/ConvStencil" "$CMAKE_FLAGS"
cmake --build "${BUILD_DIR}/ConvStencil" -j
# LoRAStencil
cmake -S "${ROOT_DIR}/third_party/LoRAStencil" -B "${BUILD_DIR}/LoRAStencil" "$CMAKE_FLAGS"
cmake --build "${BUILD_DIR}/LoRAStencil" -j
# FlashFFTStencil
mkdir -p "${BUILD_DIR}/FlashFFTStencil"
nvcc "${ROOT_DIR}/third_party/FlashFFTStencil/src/2D/2d_main.cu" -o "${BUILD_DIR}/FlashFFTStencil/2d.out" -lcufft -O3 --use_fast_math --gpu-architecture=native
nvcc "${ROOT_DIR}/third_party/FlashFFTStencil/src/3D/3d_main.cu" -o "${BUILD_DIR}/FlashFFTStencil/3d.out" -lcufft -O3 --use_fast_math --gpu-architecture=native
