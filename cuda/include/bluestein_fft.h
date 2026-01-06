/**
 * Bluestein FFT (Chirp-Z Transform) for HEALPix ring processing
 *
 * Header file declaring kernels and utility functions for both
 * forward (map2alm) and inverse (alm2map) transforms.
 */

#ifndef BLUESTEIN_FFT_H
#define BLUESTEIN_FFT_H

#include <cuda_runtime.h>
#include <cufft.h>

// ============================================================================
// Utility functions
// ============================================================================

__host__ __device__ inline int next_power_of_2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Get cached cuFFT plan (thread-safe)
cufftHandle get_cached_fft_plan(int size, int batch, cufftType type);

// ============================================================================
// Forward Bluestein (map2alm): map pixels -> Fourier coefficients Gm
// ============================================================================

template<typename T>
__global__ void bluestein_forward_pre_chirp_kernel(
    int nside, int n_maps, int n_rings, int M,
    const T* __restrict__ map_in,
    cufftDoubleComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
);

template<typename T>
__global__ void bluestein_forward_pre_chirp_kernel_f32(
    int nside, int n_maps, int n_rings, int M,
    const T* __restrict__ map_in,
    cufftComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
);

template<typename R>
__global__ void bluestein_forward_extract_gm_kernel(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,
    R* __restrict__ Gm_even_re,
    R* __restrict__ Gm_even_im,
    R* __restrict__ Gm_odd_re,
    R* __restrict__ Gm_odd_im
);

template<typename R>
__global__ void bluestein_forward_extract_gm_kernel_f32(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    R* __restrict__ Gm_even_re,
    R* __restrict__ Gm_even_im,
    R* __restrict__ Gm_odd_re,
    R* __restrict__ Gm_odd_im
);

// ============================================================================
// Inverse Bluestein (alm2map): Fourier coefficients Fmy -> map pixels
// ============================================================================

template<typename R>
__global__ void bluestein_inverse_pre_chirp_kernel(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const R* __restrict__ Fmy_re,
    const R* __restrict__ Fmy_im,
    cufftDoubleComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
);

template<typename R>
__global__ void bluestein_inverse_pre_chirp_kernel_f32(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const R* __restrict__ Fmy_re,
    const R* __restrict__ Fmy_im,
    cufftComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
);

template<typename T>
__global__ void bluestein_inverse_extract_map_kernel(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,
    T* __restrict__ map_out
);

template<typename T>
__global__ void bluestein_inverse_extract_map_kernel_f32(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    T* __restrict__ map_out
);

// ============================================================================
// Shared kernels: Conjugate chirp and pointwise multiplication
// ============================================================================

__global__ void bluestein_compute_conj_chirp_kernel_v2(
    int nside, int M,
    cufftDoubleComplex* __restrict__ conj_chirp_fft
);

__global__ void bluestein_compute_conj_chirp_kernel_f32_v2(
    int nside, int M,
    cufftComplex* __restrict__ conj_chirp_fft
);

__global__ void bluestein_pointwise_mult_kernel_v2(
    int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    cufftDoubleComplex* __restrict__ fft_data,
    const cufftDoubleComplex* __restrict__ conj_chirp_fft
);

__global__ void bluestein_pointwise_mult_kernel_f32_v2(
    int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    cufftComplex* __restrict__ fft_data,
    const cufftComplex* __restrict__ conj_chirp_fft
);

#endif // BLUESTEIN_FFT_H
