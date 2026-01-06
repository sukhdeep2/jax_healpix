/**
 * Unified Bluestein FFT (Chirp-Z Transform) Module
 *
 * Internal header for the modularized Bluestein FFT implementation.
 * Provides forward DFT (map→Gm) and inverse DFT (Fmy→map) via the
 * Bluestein algorithm, which converts arbitrary-size DFT to convolution.
 */

#ifndef BLUESTEIN_FFT_CUH
#define BLUESTEIN_FFT_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include "../../include/precision_traits.cuh"
#include "../../include/ring_geometry.cuh"

// ============================================================================
// Utility functions
// ============================================================================

__host__ __device__ inline int next_power_of_2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

__host__ __device__ inline int get_bluestein_fft_size(int max_ring_size) {
    return next_power_of_2(2 * max_ring_size - 1);
}

// ============================================================================
// cuFFT Plan Cache
// ============================================================================

cufftHandle get_cached_cufft_plan(int size, int batch, cufftType type);
cufftHandle get_cached_fft_plan(int size, int batch, cufftType type);  // Legacy alias
void clear_cufft_plan_cache();

// ============================================================================
// Conjugate Chirp Computation - precision-specific kernels
// ============================================================================

__global__ void bluestein_compute_conj_chirp_f64(
    int nside, int M,
    cufftDoubleComplex* __restrict__ conj_chirp_fft
);

__global__ void bluestein_compute_conj_chirp_f32(
    int nside, int M,
    cufftComplex* __restrict__ conj_chirp_fft
);

// ============================================================================
// Pointwise Multiplication - precision-specific kernels
// ============================================================================

__global__ void bluestein_pointwise_mult_f64(
    int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    cufftDoubleComplex* __restrict__ fft_data,
    const cufftDoubleComplex* __restrict__ conj_chirp_fft
);

__global__ void bluestein_pointwise_mult_f32(
    int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    cufftComplex* __restrict__ fft_data,
    const cufftComplex* __restrict__ conj_chirp_fft
);

// ============================================================================
// Forward Bluestein: map pixels -> Gm
// ============================================================================

// Double precision map input, double complex FFT
__global__ void bluestein_forward_pre_chirp_f64_f64(
    int nside, int n_maps, int n_rings, int M,
    const double* __restrict__ map_in,
    cufftDoubleComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
);

// Float precision map input, double complex FFT
__global__ void bluestein_forward_pre_chirp_f32_f64(
    int nside, int n_maps, int n_rings, int M,
    const float* __restrict__ map_in,
    cufftDoubleComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
);

// Double precision map input, float complex FFT
__global__ void bluestein_forward_pre_chirp_f64_f32(
    int nside, int n_maps, int n_rings, int M,
    const double* __restrict__ map_in,
    cufftComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
);

// Float precision map input, float complex FFT
__global__ void bluestein_forward_pre_chirp_f32_f32(
    int nside, int n_maps, int n_rings, int M,
    const float* __restrict__ map_in,
    cufftComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
);

// Extract Gm from double complex IFFT, double output
__global__ void bluestein_forward_extract_gm_f64_f64(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,
    double* __restrict__ Gm_even_re,
    double* __restrict__ Gm_even_im,
    double* __restrict__ Gm_odd_re,
    double* __restrict__ Gm_odd_im
);

// Extract Gm from double complex IFFT, float output
__global__ void bluestein_forward_extract_gm_f32_f64(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,
    float* __restrict__ Gm_even_re,
    float* __restrict__ Gm_even_im,
    float* __restrict__ Gm_odd_re,
    float* __restrict__ Gm_odd_im
);

// Extract Gm from float complex IFFT, double output
__global__ void bluestein_forward_extract_gm_f64_f32(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    double* __restrict__ Gm_even_re,
    double* __restrict__ Gm_even_im,
    double* __restrict__ Gm_odd_re,
    double* __restrict__ Gm_odd_im
);

// Extract Gm from float complex IFFT, float output
__global__ void bluestein_forward_extract_gm_f32_f32(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    float* __restrict__ Gm_even_re,
    float* __restrict__ Gm_even_im,
    float* __restrict__ Gm_odd_re,
    float* __restrict__ Gm_odd_im
);

// ============================================================================
// Inverse Bluestein: Fmy -> map pixels
// ============================================================================

// Double Fmy input, double complex FFT
__global__ void bluestein_inverse_pre_chirp_f64_f64(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const double* __restrict__ Fmy_re,
    const double* __restrict__ Fmy_im,
    cufftDoubleComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
);

// Float Fmy input, double complex FFT
__global__ void bluestein_inverse_pre_chirp_f32_f64(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const float* __restrict__ Fmy_re,
    const float* __restrict__ Fmy_im,
    cufftDoubleComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
);

// Double Fmy input, float complex FFT
__global__ void bluestein_inverse_pre_chirp_f64_f32(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const double* __restrict__ Fmy_re,
    const double* __restrict__ Fmy_im,
    cufftComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
);

// Float Fmy input, float complex FFT
__global__ void bluestein_inverse_pre_chirp_f32_f32(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const float* __restrict__ Fmy_re,
    const float* __restrict__ Fmy_im,
    cufftComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
);

// Extract map from double complex IFFT, double output
__global__ void bluestein_inverse_extract_map_f64_f64(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,
    double* __restrict__ map_out
);

// Extract map from double complex IFFT, float output
__global__ void bluestein_inverse_extract_map_f32_f64(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,
    float* __restrict__ map_out
);

// Extract map from float complex IFFT, double output
__global__ void bluestein_inverse_extract_map_f64_f32(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    double* __restrict__ map_out
);

// Extract map from float complex IFFT, float output
__global__ void bluestein_inverse_extract_map_f32_f32(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    float* __restrict__ map_out
);

#endif // BLUESTEIN_FFT_CUH
