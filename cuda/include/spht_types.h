#ifndef SPHT_TYPES_H
#define SPHT_TYPES_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

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

#endif // SPHT_TYPES_H
