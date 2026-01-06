/**
 * Bluestein FFT (Chirp-Z Transform) for HEALPix ring processing
 *
 * Reusable implementation for both map2alm (forward DFT) and alm2map (inverse DFT).
 *
 * Bluestein algorithm converts arbitrary-size DFT to convolution:
 *   DFT:  X[k] = sum_n x[n] * exp(-2πi*k*n/N)
 *   IDFT: x[n] = (1/N) * sum_k X[k] * exp(+2πi*k*n/N)
 *
 * Using identity: k*n = (k² + n² - (k-n)²) / 2
 *   exp(-2πi*k*n/N) = exp(-πi*k²/N) * exp(-πi*n²/N) * exp(+πi*(k-n)²/N)
 *
 * This transforms DFT into:
 *   1. Pre-chirp: multiply x[n] by exp(-πi*n²/N)
 *   2. Convolution with chirp sequence exp(+πi*j²/N)
 *   3. Post-chirp: multiply result by exp(-πi*k²/N)
 *
 * For IDFT (synthesis), the signs are flipped and we divide by N.
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include "../include/spht_types.h"
#include "../include/bluestein_fft.h"
#include <stdio.h>
#include <type_traits>
#include <unordered_map>
#include <mutex>

// ============================================================================
// cuFFT plan cache (shared across all uses)
// ============================================================================

struct PlanKey {
    int size;
    int batch;
    cufftType type;

    bool operator==(const PlanKey& other) const {
        return size == other.size && batch == other.batch && type == other.type;
    }
};

struct PlanKeyHash {
    size_t operator()(const PlanKey& k) const {
        return std::hash<int>()(k.size) ^ (std::hash<int>()(k.batch) << 8) ^
               (std::hash<int>()(static_cast<int>(k.type)) << 16);
    }
};

static std::unordered_map<PlanKey, cufftHandle, PlanKeyHash> g_plan_cache;
static std::mutex g_plan_mutex;

cufftHandle get_cached_fft_plan(int size, int batch, cufftType type) {
    PlanKey key{size, batch, type};

    std::lock_guard<std::mutex> lock(g_plan_mutex);

    auto it = g_plan_cache.find(key);
    if (it != g_plan_cache.end()) {
        return it->second;
    }

    cufftHandle plan;
    cufftResult result = cufftPlan1d(&plan, size, type, batch);
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT plan creation failed: size=%d, batch=%d, error=%d\n",
                size, batch, result);
        return 0;
    }

    g_plan_cache[key] = plan;
    return plan;
}

// ============================================================================
// Ring geometry computation (shared helper)
// ============================================================================

template<typename T>
__device__ __forceinline__ void compute_ring_geometry(
    int ring_idx, int nside,
    T* cos_theta, T* sin_theta, T* phi_0, int* n_pixels
) {
    int ring_i = ring_idx + 1;  // 1-indexed
    T cos_th, sin_th, phi0;
    int npix;

    if (ring_i < nside) {
        // North polar cap
        T i2_3n2 = T(ring_i * ring_i) / T(3.0 * nside * nside);
        cos_th = T(1.0) - i2_3n2;
        sin_th = sqrt(T(1.0) - cos_th * cos_th);
        phi0 = T(M_PI) / T(2.0 * ring_i) * T(0.5);
        npix = 4 * ring_i;
    } else if (ring_i > 3 * nside) {
        // South polar cap
        int mirror_i = 4 * nside - ring_i;
        T i2_3n2 = T(mirror_i * mirror_i) / T(3.0 * nside * nside);
        cos_th = -(T(1.0) - i2_3n2);
        sin_th = sqrt(T(1.0) - cos_th * cos_th);
        phi0 = T(M_PI) / T(2.0 * mirror_i) * T(0.5);
        npix = 4 * mirror_i;
    } else {
        // Equatorial belt
        cos_th = T(4.0 / 3.0) - T(2.0 * ring_i) / T(3.0 * nside);
        sin_th = sqrt(T(1.0) - cos_th * cos_th);
        int s = (ring_i % 2 == 0) ? 1 : 2;
        phi0 = T(M_PI) / T(2.0 * nside) * T(1.0 - s / 2.0);
        npix = 4 * nside;
    }

    *cos_theta = cos_th;
    *sin_theta = sin_th;
    *phi_0 = phi0;
    *n_pixels = npix;
}

// ============================================================================
// Forward Bluestein (map2alm): map pixels -> Fourier coefficients Gm
// ============================================================================

// Step 1: Pre-chirp - multiply input by exp(-πi*n²/N), zero-pad to M
template<typename T>
__global__ void bluestein_forward_pre_chirp_kernel(
    int nside, int n_maps, int n_rings, int M,
    const T* __restrict__ map_in,           // [n_maps, n_rings, max_pix]
    cufftDoubleComplex* __restrict__ chirped_out,  // [n_maps, n_rings, M]
    int* __restrict__ ring_sizes_out        // [n_rings]
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    // Get ring size
    double cos_th, sin_th, phi0;
    int N;
    compute_ring_geometry<double>(ring_idx, nside, &cos_th, &sin_th, &phi0, &N);

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    const T* ring_data = map_in + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;
    cufftDoubleComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;

    double pi_over_N = M_PI / double(N);

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

// Float32 version
template<typename T>
__global__ void bluestein_forward_pre_chirp_kernel_f32(
    int nside, int n_maps, int n_rings, int M,
    const T* __restrict__ map_in,
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

    const T* ring_data = map_in + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;
    cufftComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;

    float pi_over_N = float(M_PI) / float(N);

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

// ============================================================================
// Inverse Bluestein (alm2map): Fourier coefficients Fmy -> map pixels
// ============================================================================

// Step 1: Pre-chirp for inverse - multiply Fmy by exp(+πi*m²/N), zero-pad to M
// Note: For IDFT, the chirp sign is positive
template<typename R>
__global__ void bluestein_inverse_pre_chirp_kernel(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const R* __restrict__ Fmy_re,           // [n_maps, lp1, n_north_rings]
    const R* __restrict__ Fmy_im,
    cufftDoubleComplex* __restrict__ chirped_out,  // [n_maps, n_rings, M]
    int* __restrict__ ring_sizes_out        // [n_rings]
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    // Get ring size and phi0
    double cos_th, sin_th, phi0;
    int N;
    compute_ring_geometry<double>(ring_idx, nside, &cos_th, &sin_th, &phi0, &N);

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    // Determine if this is north or south ring and get Fmy index
    int north_ring = (ring_idx < n_north_rings) ? ring_idx : (n_rings - 1 - ring_idx);
    bool is_south = (ring_idx >= n_north_rings);

    cufftDoubleComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;

    double pi_over_N = M_PI / double(N);

    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        cufftDoubleComplex val;
        if (m <= l_max) {
            // Load Fmy value
            size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            double fmy_re = double(Fmy_re[idx]);
            double fmy_im = double(Fmy_im[idx]);

            // Apply phase correction: multiply by exp(+i * m * phi0)
            double phase_angle = double(m) * phi0;
            double phase_c, phase_s;
            sincos(phase_angle, &phase_s, &phase_c);
            double fmy_re_corr = fmy_re * phase_c - fmy_im * phase_s;
            double fmy_im_corr = fmy_re * phase_s + fmy_im * phase_c;

            // Pre-chirp for IDFT: multiply by exp(+πi*m²/N)
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

// Float32 version
template<typename R>
__global__ void bluestein_inverse_pre_chirp_kernel_f32(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const R* __restrict__ Fmy_re,
    const R* __restrict__ Fmy_im,
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

    float pi_over_N = float(M_PI) / float(N);

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

// ============================================================================
// Shared kernels: Conjugate chirp computation and pointwise multiplication
// ============================================================================

// Compute conjugate chirp for convolution: h[j] = exp(+πi*j²/N) with wrap-around
__global__ void bluestein_compute_conj_chirp_kernel_v2(
    int nside, int M,
    cufftDoubleComplex* __restrict__ conj_chirp_fft  // [nside, M]
) {
    int size_idx = blockIdx.x;  // 0 = size 4, 1 = size 8, ..., nside-1 = size 4*nside
    int N = 4 * (size_idx + 1);

    if (N > 4 * nside) return;

    cufftDoubleComplex* chirp = conj_chirp_fft + size_idx * M;
    double pi_over_N = M_PI / double(N);

    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        cufftDoubleComplex val;
        int j_eff;

        if (j < N) {
            j_eff = j;
        } else if (j >= M - N + 1) {
            j_eff = M - j;  // Wrap-around
        } else {
            val.x = 0.0;
            val.y = 0.0;
            chirp[j] = val;
            continue;
        }

        double angle = pi_over_N * double(j_eff) * double(j_eff);
        double c, s;
        sincos(angle, &s, &c);
        val.x = c;
        val.y = s;
        chirp[j] = val;
    }
}

// Float32 version
__global__ void bluestein_compute_conj_chirp_kernel_f32_v2(
    int nside, int M,
    cufftComplex* __restrict__ conj_chirp_fft
) {
    int size_idx = blockIdx.x;
    int N = 4 * (size_idx + 1);

    if (N > 4 * nside) return;

    cufftComplex* chirp = conj_chirp_fft + size_idx * M;
    float pi_over_N = float(M_PI) / float(N);

    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        cufftComplex val;
        int j_eff;

        if (j < N) {
            j_eff = j;
        } else if (j >= M - N + 1) {
            j_eff = M - j;
        } else {
            val.x = 0.0f;
            val.y = 0.0f;
            chirp[j] = val;
            continue;
        }

        float angle = pi_over_N * float(j_eff) * float(j_eff);
        float c, s;
        sincosf(angle, &s, &c);
        val.x = c;
        val.y = s;
        chirp[j] = val;
    }
}

// Pointwise multiplication with ring-specific conjugate chirp
__global__ void bluestein_pointwise_mult_kernel_v2(
    int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    cufftDoubleComplex* __restrict__ fft_data,       // [n_maps, n_rings, M] in-place
    const cufftDoubleComplex* __restrict__ conj_chirp_fft  // [nside, M]
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    int size_idx = (N / 4) - 1;

    cufftDoubleComplex* data = fft_data + (size_t)map_idx * n_rings * M + ring_idx * M;
    const cufftDoubleComplex* chirp = conj_chirp_fft + size_idx * M;

    for (int k = threadIdx.x; k < M; k += blockDim.x) {
        cufftDoubleComplex d = data[k];
        cufftDoubleComplex h = chirp[k];
        cufftDoubleComplex result;
        result.x = d.x * h.x - d.y * h.y;
        result.y = d.x * h.y + d.y * h.x;
        data[k] = result;
    }
}

// Float32 version
__global__ void bluestein_pointwise_mult_kernel_f32_v2(
    int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    cufftComplex* __restrict__ fft_data,
    const cufftComplex* __restrict__ conj_chirp_fft
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    int size_idx = (N / 4) - 1;

    cufftComplex* data = fft_data + (size_t)map_idx * n_rings * M + ring_idx * M;
    const cufftComplex* chirp = conj_chirp_fft + size_idx * M;

    for (int k = threadIdx.x; k < M; k += blockDim.x) {
        cufftComplex d = data[k];
        cufftComplex h = chirp[k];
        cufftComplex result;
        result.x = d.x * h.x - d.y * h.y;
        result.y = d.x * h.y + d.y * h.x;
        data[k] = result;
    }
}

// ============================================================================
// Post-processing kernels
// ============================================================================

// Forward: Extract Gm from IFFT result with post-chirp and phase correction
template<typename R>
__global__ void bluestein_forward_extract_gm_kernel(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,  // [n_maps, n_rings, M]
    R* __restrict__ Gm_even_re,
    R* __restrict__ Gm_even_im,
    R* __restrict__ Gm_odd_re,
    R* __restrict__ Gm_odd_im
) {
    int north_ring = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;

    if (north_ring >= n_north_rings || map_idx >= n_maps) return;

    int south_ring = n_rings - 1 - north_ring;
    bool is_equator = (north_ring == 2 * nside - 1);

    int N_north = ring_sizes[north_ring];
    int N_south = is_equator ? 0 : ring_sizes[south_ring];

    const cufftDoubleComplex* ifft_north = ifft_data +
        (size_t)map_idx * n_rings * M + north_ring * M;
    const cufftDoubleComplex* ifft_south = is_equator ? nullptr :
        (ifft_data + (size_t)map_idx * n_rings * M + south_ring * M);

    double pi_over_N_north = M_PI / double(N_north);
    double pi_over_N_south = is_equator ? 0.0 : M_PI / double(N_south);
    double inv_M = 1.0 / double(M);

    // Compute phi_0 for phase correction
    double phi0_north, phi0_south = 0.0;
    {
        int ring_i = north_ring + 1;
        if (ring_i < nside) {
            phi0_north = M_PI / (2.0 * ring_i) * 0.5;
        } else if (ring_i <= 3 * nside) {
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0_north = M_PI / (2.0 * nside) * (1.0 - s / 2.0);
        } else {
            int mirror_i = 4 * nside - ring_i;
            phi0_north = M_PI / (2.0 * mirror_i) * 0.5;
        }
    }
    if (!is_equator) {
        int ring_i = south_ring + 1;
        if (ring_i < nside) {
            phi0_south = M_PI / (2.0 * ring_i) * 0.5;
        } else if (ring_i <= 3 * nside) {
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0_south = M_PI / (2.0 * nside) * (1.0 - s / 2.0);
        } else {
            int mirror_i = 4 * nside - ring_i;
            phi0_south = M_PI / (2.0 * mirror_i) * 0.5;
        }
    }

    for (int m = threadIdx.x; m <= l_max; m += blockDim.x) {
        // North ring
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

        // South ring
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

        // Combine N/S
        size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
        Gm_even_re[idx] = R(gn_re + gs_re);
        Gm_even_im[idx] = R(gn_im + gs_im);
        Gm_odd_re[idx]  = R(gn_re - gs_re);
        Gm_odd_im[idx]  = R(gn_im - gs_im);
    }
}

// Float32 version
template<typename R>
__global__ void bluestein_forward_extract_gm_kernel_f32(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    R* __restrict__ Gm_even_re,
    R* __restrict__ Gm_even_im,
    R* __restrict__ Gm_odd_re,
    R* __restrict__ Gm_odd_im
) {
    int north_ring = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;

    if (north_ring >= n_north_rings || map_idx >= n_maps) return;

    int south_ring = n_rings - 1 - north_ring;
    bool is_equator = (north_ring == 2 * nside - 1);

    int N_north = ring_sizes[north_ring];
    int N_south = is_equator ? 0 : ring_sizes[south_ring];

    const cufftComplex* ifft_north = ifft_data +
        (size_t)map_idx * n_rings * M + north_ring * M;
    const cufftComplex* ifft_south = is_equator ? nullptr :
        (ifft_data + (size_t)map_idx * n_rings * M + south_ring * M);

    float pi_over_N_north = float(M_PI) / float(N_north);
    float pi_over_N_south = is_equator ? 0.0f : float(M_PI) / float(N_south);
    float inv_M = 1.0f / float(M);

    float phi0_north, phi0_south = 0.0f;
    {
        int ring_i = north_ring + 1;
        if (ring_i < nside) {
            phi0_north = float(M_PI) / float(2.0 * ring_i) * 0.5f;
        } else if (ring_i <= 3 * nside) {
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0_north = float(M_PI) / float(2.0 * nside) * (1.0f - s / 2.0f);
        } else {
            int mirror_i = 4 * nside - ring_i;
            phi0_north = float(M_PI) / float(2.0 * mirror_i) * 0.5f;
        }
    }
    if (!is_equator) {
        int ring_i = south_ring + 1;
        if (ring_i < nside) {
            phi0_south = float(M_PI) / float(2.0 * ring_i) * 0.5f;
        } else if (ring_i <= 3 * nside) {
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0_south = float(M_PI) / float(2.0 * nside) * (1.0f - s / 2.0f);
        } else {
            int mirror_i = 4 * nside - ring_i;
            phi0_south = float(M_PI) / float(2.0 * mirror_i) * 0.5f;
        }
    }

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
        Gm_even_re[idx] = R(gn_re + gs_re);
        Gm_even_im[idx] = R(gn_im + gs_im);
        Gm_odd_re[idx]  = R(gn_re - gs_re);
        Gm_odd_im[idx]  = R(gn_im - gs_im);
    }
}

// Inverse: Extract map pixels from IFFT result with post-chirp
template<typename T>
__global__ void bluestein_inverse_extract_map_kernel(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,  // [n_maps, n_rings, M]
    T* __restrict__ map_out  // [n_maps, n_rings, max_pix]
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    double pi_over_N = M_PI / double(N);
    double inv_M = 1.0 / double(M);

    const cufftDoubleComplex* ifft_ring = ifft_data +
        (size_t)map_idx * n_rings * M + ring_idx * M;
    T* map_ring = map_out + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        cufftDoubleComplex z = ifft_ring[n];
        z.x *= inv_M;
        z.y *= inv_M;

        // Post-chirp for IDFT: multiply by exp(+πi*n²/N)
        double post_angle = pi_over_N * double(n) * double(n);
        double post_c, post_s;
        sincos(post_angle, &post_s, &post_c);
        double result = z.x * post_c - z.y * post_s;

        map_ring[n] = T(result);
    }
}

// Float32 version
template<typename T>
__global__ void bluestein_inverse_extract_map_kernel_f32(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    T* __restrict__ map_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    float pi_over_N = float(M_PI) / float(N);
    float inv_M = 1.0f / float(M);

    const cufftComplex* ifft_ring = ifft_data +
        (size_t)map_idx * n_rings * M + ring_idx * M;
    T* map_ring = map_out + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        cufftComplex z = ifft_ring[n];
        z.x *= inv_M;
        z.y *= inv_M;

        float post_angle = pi_over_N * float(n) * float(n);
        float post_c, post_s;
        sincosf(post_angle, &post_s, &post_c);
        float result = z.x * post_c - z.y * post_s;

        map_ring[n] = T(result);
    }
}

// ============================================================================
// Explicit template instantiations
// ============================================================================

template __global__ void bluestein_forward_pre_chirp_kernel<double>(int, int, int, int, const double*, cufftDoubleComplex*, int*);
template __global__ void bluestein_forward_pre_chirp_kernel<float>(int, int, int, int, const float*, cufftDoubleComplex*, int*);
template __global__ void bluestein_forward_pre_chirp_kernel_f32<double>(int, int, int, int, const double*, cufftComplex*, int*);
template __global__ void bluestein_forward_pre_chirp_kernel_f32<float>(int, int, int, int, const float*, cufftComplex*, int*);

template __global__ void bluestein_inverse_pre_chirp_kernel<double>(int, int, int, int, int, int, const double*, const double*, cufftDoubleComplex*, int*);
template __global__ void bluestein_inverse_pre_chirp_kernel<float>(int, int, int, int, int, int, const float*, const float*, cufftDoubleComplex*, int*);
template __global__ void bluestein_inverse_pre_chirp_kernel_f32<double>(int, int, int, int, int, int, const double*, const double*, cufftComplex*, int*);
template __global__ void bluestein_inverse_pre_chirp_kernel_f32<float>(int, int, int, int, int, int, const float*, const float*, cufftComplex*, int*);

template __global__ void bluestein_forward_extract_gm_kernel<double>(int, int, int, int, int, int, const int*, const cufftDoubleComplex*, double*, double*, double*, double*);
template __global__ void bluestein_forward_extract_gm_kernel<float>(int, int, int, int, int, int, const int*, const cufftDoubleComplex*, float*, float*, float*, float*);
template __global__ void bluestein_forward_extract_gm_kernel_f32<double>(int, int, int, int, int, int, const int*, const cufftComplex*, double*, double*, double*, double*);
template __global__ void bluestein_forward_extract_gm_kernel_f32<float>(int, int, int, int, int, int, const int*, const cufftComplex*, float*, float*, float*, float*);

template __global__ void bluestein_inverse_extract_map_kernel<double>(int, int, int, int, const int*, const cufftDoubleComplex*, double*);
template __global__ void bluestein_inverse_extract_map_kernel<float>(int, int, int, int, const int*, const cufftDoubleComplex*, float*);
template __global__ void bluestein_inverse_extract_map_kernel_f32<double>(int, int, int, int, const int*, const cufftComplex*, double*);
template __global__ void bluestein_inverse_extract_map_kernel_f32<float>(int, int, int, int, const int*, const cufftComplex*, float*);
