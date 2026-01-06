/**
 * Bluestein FFT Forward Transform: map pixels -> Gm (Fourier coefficients)
 */

#include "bluestein_fft.cuh"
#include <stdio.h>

// ============================================================================
// Pre-chirp Kernels
// ============================================================================

// f64_f64: double map input, double complex FFT
__global__ void bluestein_forward_pre_chirp_f64_f64(
    int nside, int n_maps, int n_rings, int M,
    const double* __restrict__ map_in,
    cufftDoubleComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    double cos_th, sin_th, phi0;
    int N;
    compute_ring_geometry<double>(ring_idx, nside, &cos_th, &sin_th, &phi0, &N);

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    const double* ring_data = map_in + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;
    cufftDoubleComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;
    double pi_over_N = PrecisionTraits<double>::PI_VAL / double(N);

    for (int n = threadIdx.x; n < M; n += blockDim.x) {
        cufftDoubleComplex val;
        if (n < N) {
            double x = ring_data[n];
            double angle = -pi_over_N * double(n) * double(n);
            double c, s;
            sincos(angle, &s, &c);
            val.x = x * c;
            val.y = x * s;
        } else {
            val.x = 0.0;
            val.y = 0.0;
        }
        chirped[n] = val;
    }
}

// f32_f64: float map input, double complex FFT
__global__ void bluestein_forward_pre_chirp_f32_f64(
    int nside, int n_maps, int n_rings, int M,
    const float* __restrict__ map_in,
    cufftDoubleComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    double cos_th, sin_th, phi0;
    int N;
    compute_ring_geometry<double>(ring_idx, nside, &cos_th, &sin_th, &phi0, &N);

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    const float* ring_data = map_in + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;
    cufftDoubleComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;
    double pi_over_N = PrecisionTraits<double>::PI_VAL / double(N);

    for (int n = threadIdx.x; n < M; n += blockDim.x) {
        cufftDoubleComplex val;
        if (n < N) {
            double x = double(ring_data[n]);
            double angle = -pi_over_N * double(n) * double(n);
            double c, s;
            sincos(angle, &s, &c);
            val.x = x * c;
            val.y = x * s;
        } else {
            val.x = 0.0;
            val.y = 0.0;
        }
        chirped[n] = val;
    }
}

// f64_f32: double map input, float complex FFT
__global__ void bluestein_forward_pre_chirp_f64_f32(
    int nside, int n_maps, int n_rings, int M,
    const double* __restrict__ map_in,
    cufftComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    float cos_th, sin_th, phi0;
    int N;
    compute_ring_geometry<float>(ring_idx, nside, &cos_th, &sin_th, &phi0, &N);

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    const double* ring_data = map_in + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;
    cufftComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;
    float pi_over_N = PrecisionTraits<float>::PI_VAL / float(N);

    for (int n = threadIdx.x; n < M; n += blockDim.x) {
        cufftComplex val;
        if (n < N) {
            float x = float(ring_data[n]);
            float angle = -pi_over_N * float(n) * float(n);
            float c, s;
            sincosf(angle, &s, &c);
            val.x = x * c;
            val.y = x * s;
        } else {
            val.x = 0.0f;
            val.y = 0.0f;
        }
        chirped[n] = val;
    }
}

// f32_f32: float map input, float complex FFT
__global__ void bluestein_forward_pre_chirp_f32_f32(
    int nside, int n_maps, int n_rings, int M,
    const float* __restrict__ map_in,
    cufftComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    float cos_th, sin_th, phi0;
    int N;
    compute_ring_geometry<float>(ring_idx, nside, &cos_th, &sin_th, &phi0, &N);

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    const float* ring_data = map_in + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;
    cufftComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;
    float pi_over_N = PrecisionTraits<float>::PI_VAL / float(N);

    for (int n = threadIdx.x; n < M; n += blockDim.x) {
        cufftComplex val;
        if (n < N) {
            float x = ring_data[n];
            float angle = -pi_over_N * float(n) * float(n);
            float c, s;
            sincosf(angle, &s, &c);
            val.x = x * c;
            val.y = x * s;
        } else {
            val.x = 0.0f;
            val.y = 0.0f;
        }
        chirped[n] = val;
    }
}

// ============================================================================
// Extract Gm Kernels
// ============================================================================

// f64_f64: double complex IFFT, double Gm output
__global__ void bluestein_forward_extract_gm_f64_f64(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,
    double* __restrict__ Gm_even_re,
    double* __restrict__ Gm_even_im,
    double* __restrict__ Gm_odd_re,
    double* __restrict__ Gm_odd_im
) {
    int north_ring = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;
    if (north_ring >= n_north_rings || map_idx >= n_maps) return;

    int south_ring = n_rings - 1 - north_ring;
    bool is_equator = (north_ring == 2 * nside - 1);

    int N_north = ring_sizes[north_ring];
    int N_south = is_equator ? 0 : ring_sizes[south_ring];

    const cufftDoubleComplex* ifft_north = ifft_data + (size_t)map_idx * n_rings * M + north_ring * M;
    const cufftDoubleComplex* ifft_south = is_equator ? nullptr :
        (ifft_data + (size_t)map_idx * n_rings * M + south_ring * M);

    double pi_over_N_north = PrecisionTraits<double>::PI_VAL / double(N_north);
    double pi_over_N_south = is_equator ? 0.0 : PrecisionTraits<double>::PI_VAL / double(N_south);
    double inv_M = 1.0 / double(M);

    double phi0_north = get_ring_phi0<double>(north_ring, nside);
    double phi0_south = is_equator ? 0.0 : get_ring_phi0<double>(south_ring, nside);

    for (int m = threadIdx.x; m <= l_max; m += blockDim.x) {
        int k_north = m % N_north;
        cufftDoubleComplex z_north = ifft_north[k_north];
        z_north.x *= inv_M;
        z_north.y *= inv_M;

        double post_angle_north = -pi_over_N_north * double(k_north) * double(k_north);
        double post_c_north, post_s_north;
        sincos(post_angle_north, &post_s_north, &post_c_north);
        double gn_re_raw = z_north.x * post_c_north - z_north.y * post_s_north;
        double gn_im_raw = z_north.x * post_s_north + z_north.y * post_c_north;

        double phase_angle_north = -double(m) * phi0_north;
        double phase_c_north, phase_s_north;
        sincos(phase_angle_north, &phase_s_north, &phase_c_north);
        double gn_re = gn_re_raw * phase_c_north - gn_im_raw * phase_s_north;
        double gn_im = gn_re_raw * phase_s_north + gn_im_raw * phase_c_north;

        double gs_re = 0.0, gs_im = 0.0;
        if (!is_equator && ifft_south != nullptr) {
            int k_south = m % N_south;
            cufftDoubleComplex z_south = ifft_south[k_south];
            z_south.x *= inv_M;
            z_south.y *= inv_M;

            double post_angle_south = -pi_over_N_south * double(k_south) * double(k_south);
            double post_c_south, post_s_south;
            sincos(post_angle_south, &post_s_south, &post_c_south);
            double gs_re_raw = z_south.x * post_c_south - z_south.y * post_s_south;
            double gs_im_raw = z_south.x * post_s_south + z_south.y * post_c_south;

            double phase_angle_south = -double(m) * phi0_south;
            double phase_c_south, phase_s_south;
            sincos(phase_angle_south, &phase_s_south, &phase_c_south);
            gs_re = gs_re_raw * phase_c_south - gs_im_raw * phase_s_south;
            gs_im = gs_re_raw * phase_s_south + gs_im_raw * phase_c_south;
        }

        size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
        Gm_even_re[idx] = gn_re + gs_re;
        Gm_even_im[idx] = gn_im + gs_im;
        Gm_odd_re[idx]  = gn_re - gs_re;
        Gm_odd_im[idx]  = gn_im - gs_im;
    }
}

// f32_f64: double complex IFFT, float Gm output
__global__ void bluestein_forward_extract_gm_f32_f64(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,
    float* __restrict__ Gm_even_re,
    float* __restrict__ Gm_even_im,
    float* __restrict__ Gm_odd_re,
    float* __restrict__ Gm_odd_im
) {
    int north_ring = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;
    if (north_ring >= n_north_rings || map_idx >= n_maps) return;

    int south_ring = n_rings - 1 - north_ring;
    bool is_equator = (north_ring == 2 * nside - 1);

    int N_north = ring_sizes[north_ring];
    int N_south = is_equator ? 0 : ring_sizes[south_ring];

    const cufftDoubleComplex* ifft_north = ifft_data + (size_t)map_idx * n_rings * M + north_ring * M;
    const cufftDoubleComplex* ifft_south = is_equator ? nullptr :
        (ifft_data + (size_t)map_idx * n_rings * M + south_ring * M);

    double pi_over_N_north = PrecisionTraits<double>::PI_VAL / double(N_north);
    double pi_over_N_south = is_equator ? 0.0 : PrecisionTraits<double>::PI_VAL / double(N_south);
    double inv_M = 1.0 / double(M);

    double phi0_north = get_ring_phi0<double>(north_ring, nside);
    double phi0_south = is_equator ? 0.0 : get_ring_phi0<double>(south_ring, nside);

    for (int m = threadIdx.x; m <= l_max; m += blockDim.x) {
        int k_north = m % N_north;
        cufftDoubleComplex z_north = ifft_north[k_north];
        z_north.x *= inv_M;
        z_north.y *= inv_M;

        double post_angle_north = -pi_over_N_north * double(k_north) * double(k_north);
        double post_c_north, post_s_north;
        sincos(post_angle_north, &post_s_north, &post_c_north);
        double gn_re_raw = z_north.x * post_c_north - z_north.y * post_s_north;
        double gn_im_raw = z_north.x * post_s_north + z_north.y * post_c_north;

        double phase_angle_north = -double(m) * phi0_north;
        double phase_c_north, phase_s_north;
        sincos(phase_angle_north, &phase_s_north, &phase_c_north);
        double gn_re = gn_re_raw * phase_c_north - gn_im_raw * phase_s_north;
        double gn_im = gn_re_raw * phase_s_north + gn_im_raw * phase_c_north;

        double gs_re = 0.0, gs_im = 0.0;
        if (!is_equator && ifft_south != nullptr) {
            int k_south = m % N_south;
            cufftDoubleComplex z_south = ifft_south[k_south];
            z_south.x *= inv_M;
            z_south.y *= inv_M;

            double post_angle_south = -pi_over_N_south * double(k_south) * double(k_south);
            double post_c_south, post_s_south;
            sincos(post_angle_south, &post_s_south, &post_c_south);
            double gs_re_raw = z_south.x * post_c_south - z_south.y * post_s_south;
            double gs_im_raw = z_south.x * post_s_south + z_south.y * post_c_south;

            double phase_angle_south = -double(m) * phi0_south;
            double phase_c_south, phase_s_south;
            sincos(phase_angle_south, &phase_s_south, &phase_c_south);
            gs_re = gs_re_raw * phase_c_south - gs_im_raw * phase_s_south;
            gs_im = gs_re_raw * phase_s_south + gs_im_raw * phase_c_south;
        }

        size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
        Gm_even_re[idx] = float(gn_re + gs_re);
        Gm_even_im[idx] = float(gn_im + gs_im);
        Gm_odd_re[idx]  = float(gn_re - gs_re);
        Gm_odd_im[idx]  = float(gn_im - gs_im);
    }
}

// f64_f32: float complex IFFT, double Gm output
__global__ void bluestein_forward_extract_gm_f64_f32(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    double* __restrict__ Gm_even_re,
    double* __restrict__ Gm_even_im,
    double* __restrict__ Gm_odd_re,
    double* __restrict__ Gm_odd_im
) {
    int north_ring = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;
    if (north_ring >= n_north_rings || map_idx >= n_maps) return;

    int south_ring = n_rings - 1 - north_ring;
    bool is_equator = (north_ring == 2 * nside - 1);

    int N_north = ring_sizes[north_ring];
    int N_south = is_equator ? 0 : ring_sizes[south_ring];

    const cufftComplex* ifft_north = ifft_data + (size_t)map_idx * n_rings * M + north_ring * M;
    const cufftComplex* ifft_south = is_equator ? nullptr :
        (ifft_data + (size_t)map_idx * n_rings * M + south_ring * M);

    float pi_over_N_north = PrecisionTraits<float>::PI_VAL / float(N_north);
    float pi_over_N_south = is_equator ? 0.0f : PrecisionTraits<float>::PI_VAL / float(N_south);
    float inv_M = 1.0f / float(M);

    float phi0_north = get_ring_phi0<float>(north_ring, nside);
    float phi0_south = is_equator ? 0.0f : get_ring_phi0<float>(south_ring, nside);

    for (int m = threadIdx.x; m <= l_max; m += blockDim.x) {
        int k_north = m % N_north;
        cufftComplex z_north = ifft_north[k_north];
        z_north.x *= inv_M;
        z_north.y *= inv_M;

        float post_angle_north = -pi_over_N_north * float(k_north) * float(k_north);
        float post_c_north, post_s_north;
        sincosf(post_angle_north, &post_s_north, &post_c_north);
        float gn_re_raw = z_north.x * post_c_north - z_north.y * post_s_north;
        float gn_im_raw = z_north.x * post_s_north + z_north.y * post_c_north;

        float phase_angle_north = -float(m) * phi0_north;
        float phase_c_north, phase_s_north;
        sincosf(phase_angle_north, &phase_s_north, &phase_c_north);
        float gn_re = gn_re_raw * phase_c_north - gn_im_raw * phase_s_north;
        float gn_im = gn_re_raw * phase_s_north + gn_im_raw * phase_c_north;

        float gs_re = 0.0f, gs_im = 0.0f;
        if (!is_equator && ifft_south != nullptr) {
            int k_south = m % N_south;
            cufftComplex z_south = ifft_south[k_south];
            z_south.x *= inv_M;
            z_south.y *= inv_M;

            float post_angle_south = -pi_over_N_south * float(k_south) * float(k_south);
            float post_c_south, post_s_south;
            sincosf(post_angle_south, &post_s_south, &post_c_south);
            float gs_re_raw = z_south.x * post_c_south - z_south.y * post_s_south;
            float gs_im_raw = z_south.x * post_s_south + z_south.y * post_c_south;

            float phase_angle_south = -float(m) * phi0_south;
            float phase_c_south, phase_s_south;
            sincosf(phase_angle_south, &phase_s_south, &phase_c_south);
            gs_re = gs_re_raw * phase_c_south - gs_im_raw * phase_s_south;
            gs_im = gs_re_raw * phase_s_south + gs_im_raw * phase_c_south;
        }

        size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
        Gm_even_re[idx] = double(gn_re + gs_re);
        Gm_even_im[idx] = double(gn_im + gs_im);
        Gm_odd_re[idx]  = double(gn_re - gs_re);
        Gm_odd_im[idx]  = double(gn_im - gs_im);
    }
}

// f32_f32: float complex IFFT, float Gm output
__global__ void bluestein_forward_extract_gm_f32_f32(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    float* __restrict__ Gm_even_re,
    float* __restrict__ Gm_even_im,
    float* __restrict__ Gm_odd_re,
    float* __restrict__ Gm_odd_im
) {
    int north_ring = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;
    if (north_ring >= n_north_rings || map_idx >= n_maps) return;

    int south_ring = n_rings - 1 - north_ring;
    bool is_equator = (north_ring == 2 * nside - 1);

    int N_north = ring_sizes[north_ring];
    int N_south = is_equator ? 0 : ring_sizes[south_ring];

    const cufftComplex* ifft_north = ifft_data + (size_t)map_idx * n_rings * M + north_ring * M;
    const cufftComplex* ifft_south = is_equator ? nullptr :
        (ifft_data + (size_t)map_idx * n_rings * M + south_ring * M);

    float pi_over_N_north = PrecisionTraits<float>::PI_VAL / float(N_north);
    float pi_over_N_south = is_equator ? 0.0f : PrecisionTraits<float>::PI_VAL / float(N_south);
    float inv_M = 1.0f / float(M);

    float phi0_north = get_ring_phi0<float>(north_ring, nside);
    float phi0_south = is_equator ? 0.0f : get_ring_phi0<float>(south_ring, nside);

    for (int m = threadIdx.x; m <= l_max; m += blockDim.x) {
        int k_north = m % N_north;
        cufftComplex z_north = ifft_north[k_north];
        z_north.x *= inv_M;
        z_north.y *= inv_M;

        float post_angle_north = -pi_over_N_north * float(k_north) * float(k_north);
        float post_c_north, post_s_north;
        sincosf(post_angle_north, &post_s_north, &post_c_north);
        float gn_re_raw = z_north.x * post_c_north - z_north.y * post_s_north;
        float gn_im_raw = z_north.x * post_s_north + z_north.y * post_c_north;

        float phase_angle_north = -float(m) * phi0_north;
        float phase_c_north, phase_s_north;
        sincosf(phase_angle_north, &phase_s_north, &phase_c_north);
        float gn_re = gn_re_raw * phase_c_north - gn_im_raw * phase_s_north;
        float gn_im = gn_re_raw * phase_s_north + gn_im_raw * phase_c_north;

        float gs_re = 0.0f, gs_im = 0.0f;
        if (!is_equator && ifft_south != nullptr) {
            int k_south = m % N_south;
            cufftComplex z_south = ifft_south[k_south];
            z_south.x *= inv_M;
            z_south.y *= inv_M;

            float post_angle_south = -pi_over_N_south * float(k_south) * float(k_south);
            float post_c_south, post_s_south;
            sincosf(post_angle_south, &post_s_south, &post_c_south);
            float gs_re_raw = z_south.x * post_c_south - z_south.y * post_s_south;
            float gs_im_raw = z_south.x * post_s_south + z_south.y * post_c_south;

            float phase_angle_south = -float(m) * phi0_south;
            float phase_c_south, phase_s_south;
            sincosf(phase_angle_south, &phase_s_south, &phase_c_south);
            gs_re = gs_re_raw * phase_c_south - gs_im_raw * phase_s_south;
            gs_im = gs_re_raw * phase_s_south + gs_im_raw * phase_c_south;
        }

        size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
        Gm_even_re[idx] = gn_re + gs_re;
        Gm_even_im[idx] = gn_im + gs_im;
        Gm_odd_re[idx]  = gn_re - gs_re;
        Gm_odd_im[idx]  = gn_im - gs_im;
    }
}
