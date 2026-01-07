#ifndef SPHT_TYPES_H
#define SPHT_TYPES_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// bf16 support (requires sm_80+ / Ampere)
#if defined(__CUDACC__)
#include <cuda_bf16.h>
#define SPHT_HAS_BF16_HEADER 1
#else
#define SPHT_HAS_BF16_HEADER 0
#endif

// ============================================================================
// Precision and Accumulation Mode Enums
// ============================================================================

/**
 * Storage/recurrence precision options
 */
typedef enum {
    SPHT_PRECISION_F64 = 0,    // float64 (double)
    SPHT_PRECISION_F32 = 1,    // float32 (float)
    SPHT_PRECISION_BF16 = 2    // bfloat16 (requires sm_80+)
} spht_precision_t;

/**
 * Accumulation mode for Phase 2 (Ylm + sum)
 *
 * LINEAR: Fast FMA-based accumulation (default for f32/f64)
 *         - Uses standard multiply-accumulate
 *         - Best performance, may overflow for extreme dynamic range
 *
 * LOG: Log-space logsumexp accumulation (required for bf16)
 *      - Uses logsumexp for numerically stable sum
 *      - ~5-10x slower than LINEAR
 *      - Handles extreme dynamic range without overflow
 */
typedef enum {
    SPHT_ACCUM_LINEAR = 0,     // Fast FMA path (default)
    SPHT_ACCUM_LOG = 1         // Log-space path (required for bf16)
} spht_accumulation_mode_t;

// ============================================================================
// Default type (for backward compatibility)
// ============================================================================

// CRITICAL: Must use double precision throughout for accuracy
typedef double real_t;

// Complex number type
typedef struct {
    double x;  // Real part
    double y;  // Imag part
} complex_t;

// Log-space representation: value = sign * exp(log_magnitude)
typedef struct {
    real_t log_magnitude;  // log(|value|)
    int8_t sign;           // +1 or -1
} log_value_t;

// YLM storage for a batch of rings
// Shape: [l_max+1, l_max+1, n_rings]
typedef struct {
    real_t* log_ylm;       // Log magnitudes, device pointer
    int8_t* sign_ylm;      // Signs, device pointer
    int l_max;
    int n_rings;
} ylm_log_t;

// Ring geometry information
typedef struct {
    real_t* log_beta;      // log(|cos(theta)|) for each ring
    int8_t* beta_sign;     // sign of cos(theta)
    real_t* log_sin_theta; // log(sin(theta)) for each ring
    real_t* phi_0;         // Starting phi for each ring
    int* n_pixels;         // Number of pixels per ring
    int n_rings;           // 4*nside - 1
    int nside;
} ring_geometry_t;

// HEALPix map in ring-ordered format
// Shape: [n_maps, 4*nside-1, 4*nside]
typedef struct {
    real_t* data;          // Device pointer (real maps)
    complex_t* data_c;     // Device pointer (complex, for spin-2)
    int n_maps;
    int nside;
} healpix_map_t;

// alm coefficients
// Shape: [n_fields, l_max+1, l_max+1]
typedef struct {
    complex_t* data;       // Device pointer
    int n_fields;
    int l_max;
} alm_t;

// Mathematical constants (double precision)
#define PI 3.14159265358979323846
#define LOG_4PI 2.5310242469692907  // log(4*pi)

// Mathematical constants (single precision)
#define PI_F 3.14159265f
#define LOG_4PI_F 2.53102425f

// Algorithm parameters
#define RING_BATCH_SIZE 16  // Match JAX RING_ITER_SIZE
#define WARP_SIZE 32

// Precision-specific limits
#define LOG_MIN_F64 -700.0
#define LOG_MAX_F64  700.0
#define LOG_MIN_F32 -87.0f
#define LOG_MAX_F32  88.0f

// 3D array indexing: arr[i, j, k] with dimensions [D1, D2, D3]
// Row-major order (C-style)
#define INDEX_3D(i, j, k, D2, D3) ((i) * (D2) * (D3) + (j) * (D3) + (k))

// 2D array indexing
#define INDEX_2D(i, j, D2) ((i) * (D2) + (j))

// Upper triangular index for (l, m) with l_max
// Only valid for m <= l
#define INDEX_UPPER_TRI(l, m) ((l) * ((l) + 1) / 2 + (m))

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Helper macros for kernel launches
#define CEILDIV(a, b) (((a) + (b) - 1) / (b))

// ============================================================================
// bf16 Storage Formats (for memory-efficient log-space storage)
// ============================================================================

#if SPHT_HAS_BF16_HEADER

/**
 * LogMapBlock: Packed bf16 storage for real-valued maps
 *
 * Stores 16 map values in log-space with packed signs.
 * Memory: 34 bytes for 16 values = 2.125 bytes/value (vs 4 for f32)
 */
typedef struct __align__(32) {
    __nv_bfloat16 log_abs[16];  // log(|value|), 32 bytes
    uint16_t signs;              // 1 bit per value, packed
    uint16_t padding;            // Alignment padding
} LogMapBlock;

/**
 * LogAlmBlock: Packed bf16 storage for complex alm coefficients
 *
 * Stores 16 complex values in log-Cartesian form with packed signs.
 * Memory: 68 bytes for 16 complex = 4.25 bytes/complex (vs 8 for complex64)
 */
typedef struct __align__(64) {
    __nv_bfloat16 log_re[16];   // log(|real|), 32 bytes
    __nv_bfloat16 log_im[16];   // log(|imag|), 32 bytes
    uint16_t signs_re;           // 1 bit per value
    uint16_t signs_im;           // 1 bit per value
} LogAlmBlock;

/**
 * Helper: Extract sign bit from packed uint16
 */
static __device__ __forceinline__ int8_t unpack_sign(uint16_t packed, int idx) {
    return ((packed >> idx) & 1) ? int8_t(1) : int8_t(-1);
}

/**
 * Helper: Pack sign into uint16
 */
static __device__ __forceinline__ uint16_t pack_sign(uint16_t packed, int idx, int8_t sign) {
    if (sign >= 0) {
        return packed | (uint16_t(1) << idx);
    } else {
        return packed & ~(uint16_t(1) << idx);
    }
}

#endif // SPHT_HAS_BF16_HEADER

#endif // SPHT_TYPES_H
