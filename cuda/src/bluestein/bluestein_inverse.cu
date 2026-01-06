/**
 * Bluestein FFT Inverse Transform: Fmy -> map pixels
 */

#include "bluestein_fft.cuh"
#include <stdio.h>

// ============================================================================
// Pre-chirp Kernels for Inverse
// ============================================================================

// f64_f64: double Fmy input, double complex FFT
__global__ void bluestein_inverse_pre_chirp_f64_f64(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const double* __restrict__ Fmy_re,
    const double* __restrict__ Fmy_im,
    cufftDoubleComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    double cos_th, sin_th, phi0;
    int N;
    compute_ring_geometry<double>(ring_idx, nside, &cos_th, &sin_th, &phi0, &N);

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    int north_ring = (ring_idx < n_north_rings) ? ring_idx : (n_rings - 1 - ring_idx);
    cufftDoubleComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;
    double pi_over_N = PrecisionTraits<double>::PI_VAL / double(N);

    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        cufftDoubleComplex val;
        if (m <= l_max) {
            size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            double fmy_re = Fmy_re[idx];
            double fmy_im = Fmy_im[idx];

            double phase_angle = double(m) * phi0;
            double phase_c, phase_s;
            sincos(phase_angle, &phase_s, &phase_c);
            double fmy_re_corr = fmy_re * phase_c - fmy_im * phase_s;
            double fmy_im_corr = fmy_re * phase_s + fmy_im * phase_c;

            double chirp_angle = pi_over_N * double(m) * double(m);
            double chirp_c, chirp_s;
            sincos(chirp_angle, &chirp_s, &chirp_c);
            val.x = fmy_re_corr * chirp_c - fmy_im_corr * chirp_s;
            val.y = fmy_re_corr * chirp_s + fmy_im_corr * chirp_c;
        } else {
            val.x = 0.0;
            val.y = 0.0;
        }
        chirped[m] = val;
    }
}

// f32_f64: float Fmy input, double complex FFT
__global__ void bluestein_inverse_pre_chirp_f32_f64(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const float* __restrict__ Fmy_re,
    const float* __restrict__ Fmy_im,
    cufftDoubleComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    double cos_th, sin_th, phi0;
    int N;
    compute_ring_geometry<double>(ring_idx, nside, &cos_th, &sin_th, &phi0, &N);

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    int north_ring = (ring_idx < n_north_rings) ? ring_idx : (n_rings - 1 - ring_idx);
    cufftDoubleComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;
    double pi_over_N = PrecisionTraits<double>::PI_VAL / double(N);

    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        cufftDoubleComplex val;
        if (m <= l_max) {
            size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            double fmy_re = double(Fmy_re[idx]);
            double fmy_im = double(Fmy_im[idx]);

            double phase_angle = double(m) * phi0;
            double phase_c, phase_s;
            sincos(phase_angle, &phase_s, &phase_c);
            double fmy_re_corr = fmy_re * phase_c - fmy_im * phase_s;
            double fmy_im_corr = fmy_re * phase_s + fmy_im * phase_c;

            double chirp_angle = pi_over_N * double(m) * double(m);
            double chirp_c, chirp_s;
            sincos(chirp_angle, &chirp_s, &chirp_c);
            val.x = fmy_re_corr * chirp_c - fmy_im_corr * chirp_s;
            val.y = fmy_re_corr * chirp_s + fmy_im_corr * chirp_c;
        } else {
            val.x = 0.0;
            val.y = 0.0;
        }
        chirped[m] = val;
    }
}

// f64_f32: double Fmy input, float complex FFT
__global__ void bluestein_inverse_pre_chirp_f64_f32(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const double* __restrict__ Fmy_re,
    const double* __restrict__ Fmy_im,
    cufftComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    float cos_th, sin_th, phi0;
    int N;
    compute_ring_geometry<float>(ring_idx, nside, &cos_th, &sin_th, &phi0, &N);

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    int north_ring = (ring_idx < n_north_rings) ? ring_idx : (n_rings - 1 - ring_idx);
    cufftComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;
    float pi_over_N = PrecisionTraits<float>::PI_VAL / float(N);

    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        cufftComplex val;
        if (m <= l_max) {
            size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            float fmy_re = float(Fmy_re[idx]);
            float fmy_im = float(Fmy_im[idx]);

            float phase_angle = float(m) * phi0;
            float phase_c, phase_s;
            sincosf(phase_angle, &phase_s, &phase_c);
            float fmy_re_corr = fmy_re * phase_c - fmy_im * phase_s;
            float fmy_im_corr = fmy_re * phase_s + fmy_im * phase_c;

            float chirp_angle = pi_over_N * float(m) * float(m);
            float chirp_c, chirp_s;
            sincosf(chirp_angle, &chirp_s, &chirp_c);
            val.x = fmy_re_corr * chirp_c - fmy_im_corr * chirp_s;
            val.y = fmy_re_corr * chirp_s + fmy_im_corr * chirp_c;
        } else {
            val.x = 0.0f;
            val.y = 0.0f;
        }
        chirped[m] = val;
    }
}

// f32_f32: float Fmy input, float complex FFT
__global__ void bluestein_inverse_pre_chirp_f32_f32(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const float* __restrict__ Fmy_re,
    const float* __restrict__ Fmy_im,
    cufftComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    float cos_th, sin_th, phi0;
    int N;
    compute_ring_geometry<float>(ring_idx, nside, &cos_th, &sin_th, &phi0, &N);

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    int north_ring = (ring_idx < n_north_rings) ? ring_idx : (n_rings - 1 - ring_idx);
    cufftComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;
    float pi_over_N = PrecisionTraits<float>::PI_VAL / float(N);

    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        cufftComplex val;
        if (m <= l_max) {
            size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            float fmy_re = Fmy_re[idx];
            float fmy_im = Fmy_im[idx];

            float phase_angle = float(m) * phi0;
            float phase_c, phase_s;
            sincosf(phase_angle, &phase_s, &phase_c);
            float fmy_re_corr = fmy_re * phase_c - fmy_im * phase_s;
            float fmy_im_corr = fmy_re * phase_s + fmy_im * phase_c;

            float chirp_angle = pi_over_N * float(m) * float(m);
            float chirp_c, chirp_s;
            sincosf(chirp_angle, &chirp_s, &chirp_c);
            val.x = fmy_re_corr * chirp_c - fmy_im_corr * chirp_s;
            val.y = fmy_re_corr * chirp_s + fmy_im_corr * chirp_c;
        } else {
            val.x = 0.0f;
            val.y = 0.0f;
        }
        chirped[m] = val;
    }
}

// ============================================================================
// Extract Map Kernels
// ============================================================================

// f64_f64: double complex IFFT, double map output
__global__ void bluestein_inverse_extract_map_f64_f64(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,
    double* __restrict__ map_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    double pi_over_N = PrecisionTraits<double>::PI_VAL / double(N);
    double inv_M = 1.0 / double(M);

    const cufftDoubleComplex* ifft_ring = ifft_data + (size_t)map_idx * n_rings * M + ring_idx * M;
    double* map_ring = map_out + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        cufftDoubleComplex z = ifft_ring[n];
        z.x *= inv_M;
        z.y *= inv_M;

        double post_angle = pi_over_N * double(n) * double(n);
        double post_c, post_s;
        sincos(post_angle, &post_s, &post_c);
        double result = z.x * post_c - z.y * post_s;
        map_ring[n] = result;
    }
}

// f32_f64: double complex IFFT, float map output
__global__ void bluestein_inverse_extract_map_f32_f64(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,
    float* __restrict__ map_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    double pi_over_N = PrecisionTraits<double>::PI_VAL / double(N);
    double inv_M = 1.0 / double(M);

    const cufftDoubleComplex* ifft_ring = ifft_data + (size_t)map_idx * n_rings * M + ring_idx * M;
    float* map_ring = map_out + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        cufftDoubleComplex z = ifft_ring[n];
        z.x *= inv_M;
        z.y *= inv_M;

        double post_angle = pi_over_N * double(n) * double(n);
        double post_c, post_s;
        sincos(post_angle, &post_s, &post_c);
        double result = z.x * post_c - z.y * post_s;
        map_ring[n] = float(result);
    }
}

// f64_f32: float complex IFFT, double map output
__global__ void bluestein_inverse_extract_map_f64_f32(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    double* __restrict__ map_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    float pi_over_N = PrecisionTraits<float>::PI_VAL / float(N);
    float inv_M = 1.0f / float(M);

    const cufftComplex* ifft_ring = ifft_data + (size_t)map_idx * n_rings * M + ring_idx * M;
    double* map_ring = map_out + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        cufftComplex z = ifft_ring[n];
        z.x *= inv_M;
        z.y *= inv_M;

        float post_angle = pi_over_N * float(n) * float(n);
        float post_c, post_s;
        sincosf(post_angle, &post_s, &post_c);
        float result = z.x * post_c - z.y * post_s;
        map_ring[n] = double(result);
    }
}

// f32_f32: float complex IFFT, float map output
__global__ void bluestein_inverse_extract_map_f32_f32(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    float* __restrict__ map_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    float pi_over_N = PrecisionTraits<float>::PI_VAL / float(N);
    float inv_M = 1.0f / float(M);

    const cufftComplex* ifft_ring = ifft_data + (size_t)map_idx * n_rings * M + ring_idx * M;
    float* map_ring = map_out + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        cufftComplex z = ifft_ring[n];
        z.x *= inv_M;
        z.y *= inv_M;

        float post_angle = pi_over_N * float(n) * float(n);
        float post_c, post_s;
        sincosf(post_angle, &post_s, &post_c);
        float result = z.x * post_c - z.y * post_s;
        map_ring[n] = result;
    }
}
