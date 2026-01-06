#!/bin/bash
# Build script for SPHT CUDA library using nvcc directly

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
SRC_DIR="${SCRIPT_DIR}/src"
INCLUDE_DIR="${SCRIPT_DIR}/include"

# CUDA compiler
NVCC=nvcc

# Compiler flags
NVCC_FLAGS="-O3 --use_fast_math -Xcompiler -fPIC -lineinfo"
NVCC_FLAGS="$NVCC_FLAGS -I${INCLUDE_DIR}"

# Architecture flags (adjust for your GPU)
# Detected: compute capability 8.6 (Ampere - RTX 3080/3090)
ARCH_FLAGS="-gencode arch=compute_86,code=sm_86"

# Parse arguments
DEBUG=0
CLEAN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG=1
            NVCC_FLAGS="-g -G -O0 -Xcompiler -fPIC -I${INCLUDE_DIR}"
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--debug] [--clean]"
            exit 1
            ;;
    esac
done

# Clean if requested
if [[ $CLEAN -eq 1 ]]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"

echo "Building SPHT CUDA library..."
echo "  Source dir: ${SRC_DIR}"
echo "  Build dir: ${BUILD_DIR}"
echo ""

# Source files
SOURCES=(
    "ylm_recurrence.cu"
    "ring_integrate.cu"
    "alm2map.cu"
    "map2alm.cu"
    "map2alm_v2.cu"    # FFT + cuBLAS version
    "map2alm_v3.cu"    # Fused per-ring parallel version
    "map2alm_v4.cu"    # Tiled version for large nside
    "map2alm_v5.cu"    # Multi-precision templated version
    "map2alm_v6.cu"    # Optimal warp-per-m, NO atomics
    "alm2map_v5.cu"    # Multi-precision synthesis version
    "alm2map_v6.cu"    # Optimal warp-per-m synthesis (matching map2alm_v6)
    "fft_gm.cu"        # FFT-based Gm
    "spht_api.cu"
)

# Compile each source file to object
OBJECTS=""
for src in "${SOURCES[@]}"; do
    obj="${BUILD_DIR}/${src%.cu}.o"
    echo "Compiling ${src}..."
    ${NVCC} ${NVCC_FLAGS} ${ARCH_FLAGS} -dc -o "${obj}" "${SRC_DIR}/${src}"
    OBJECTS="${OBJECTS} ${obj}"
done

# Device link with -Xcompiler -fPIC
echo ""
echo "Device linking..."
${NVCC} ${ARCH_FLAGS} -Xcompiler -fPIC -dlink -o "${BUILD_DIR}/device_link.o" ${OBJECTS}

# Create shared library
echo ""
echo "Creating shared library..."
${NVCC} -shared -Xcompiler -fPIC -o "${BUILD_DIR}/libspht_cuda.so" \
    ${OBJECTS} "${BUILD_DIR}/device_link.o" \
    -lcudart -lcufft -lcublas

echo ""
echo "Building test executables..."

# Build test_ylm
echo "Compiling test_ylm.cu..."
${NVCC} ${NVCC_FLAGS} ${ARCH_FLAGS} -o "${BUILD_DIR}/test_ylm" \
    "${SCRIPT_DIR}/tests/test_ylm.cu" \
    -L"${BUILD_DIR}" -lspht_cuda -Xlinker -rpath,"${BUILD_DIR}"

# Build test_transforms
echo "Compiling test_transforms.cu..."
${NVCC} ${NVCC_FLAGS} ${ARCH_FLAGS} -o "${BUILD_DIR}/test_transforms" \
    "${SCRIPT_DIR}/tests/test_transforms.cu" \
    -L"${BUILD_DIR}" -lspht_cuda -Xlinker -rpath,"${BUILD_DIR}"

echo ""
echo "Build complete!"
echo ""
echo "Library: ${BUILD_DIR}/libspht_cuda.so"
echo "Tests:"
echo "  ${BUILD_DIR}/test_ylm"
echo "  ${BUILD_DIR}/test_transforms"
echo ""
echo "Run tests with:"
echo "  ${BUILD_DIR}/test_ylm"
echo "  ${BUILD_DIR}/test_transforms"
