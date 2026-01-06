cmake_minimum_required(VERSION 3.18)
project(spht_cuda CUDA CXX)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (adjust for your GPU)
# Common architectures:
# - 60: Pascal (GTX 1080, Tesla P100)
# - 70: Volta (Tesla V100)
# - 75: Turing (RTX 2080)
# - 80: Ampere (A100, RTX 3090)
# - 86: Ampere (RTX 3080)
# - 89: Ada (RTX 4090)
# - 90: Hopper (H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/bluestein)

# Source files - organized by module
set(BLUESTEIN_SOURCES
    src/bluestein/bluestein_core.cu
    src/bluestein/bluestein_forward.cu
    src/bluestein/bluestein_inverse.cu
)

set(CORE_SOURCES
    src/ylm_recurrence.cu
    src/map2alm_v6.cu
    src/alm2map_v6.cu
    src/alm2cl.cu
    src/spht_api.cu
)

set(CUDA_SOURCES
    ${BLUESTEIN_SOURCES}
    ${CORE_SOURCES}
)

# Build shared library
add_library(spht_cuda SHARED ${CUDA_SOURCES})

# Link cuFFT and cuBLAS
target_link_libraries(spht_cuda CUDA::cudart CUDA::cufft CUDA::cublas)

# Set compiler flags
target_compile_options(spht_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xcompiler -fPIC
        -lineinfo
    >
)

# Enable position independent code
set_target_properties(spht_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Debug build options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(spht_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
    )
    add_definitions(-DDEBUG)
endif()

# Install rules
install(TARGETS spht_cuda
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include/spht_cuda
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# Build test executable (optional)
option(BUILD_TESTS "Build test programs" ON)

if(BUILD_TESTS)
    add_executable(test_ylm tests/test_ylm.cu)
    target_link_libraries(test_ylm spht_cuda CUDA::cudart)

    add_executable(test_transforms tests/test_transforms.cu)
    target_link_libraries(test_transforms spht_cuda CUDA::cudart)
endif()
