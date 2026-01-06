/**
 * map2alm_v5: Multi-precision batched implementation
 *
 * Supports float64 (double), float32 (float), and float16 (half).
 * Uses C++ templates for clean precision switching.
 *
 * Note: For high l_max, float64 is recommended. Lower precision may
 * cause numerical instability in the Ylm recurrence.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For half precision
#include "../include/spht_types.h"
#include <stdio.h>
#include <type_traits>

// Batch size for loading ring pixels
#define PIXEL_BATCH_SIZE 2048
#define MAX_M_PER_THREAD 64

// ============================================================================
// Type traits for precision handling
// ============================================================================

template<typename T>
struct PrecisionTraits;

template<>
struct PrecisionTraits<double> {
    using real_type = double;
    using real2_type = double2;
    using accum_type = double;  // Accumulation type
    static constexpr double PI_VAL = M_PI;
    static __device__ __forceinline__ double sqrt_impl(double x) { return sqrt(x); }
    static __device__ __forceinline__ void sincos_impl(double x, double* s, double* c) { sincos(x, s, c); }
    static __device__ __forceinline__ double2 make_real2(double x, double y) { return make_double2(x, y); }
};

template<>
struct PrecisionTraits<float> {
    using real_type = float;
    using real2_type = float2;
    using accum_type = float;
    static constexpr float PI_VAL = 3.14159265f;
    static __device__ __forceinline__ float sqrt_impl(float x) { return sqrtf(x); }
    static __device__ __forceinline__ void sincos_impl(float x, float* s, float* c) { sincosf(x, s, c); }
    static __device__ __forceinline__ float2 make_real2(float x, float y) { return make_float2(x, y); }
};

// Half precision - uses float for intermediate computations
template<>
struct PrecisionTraits<half> {
    using real_type = half;
    using real2_type = half2;
    using accum_type = float;  // Accumulate in float for stability
    static constexpr float PI_VAL = 3.14159265f;
    static __device__ __forceinline__ float sqrt_impl(float x) { return sqrtf(x); }
    static __device__ __forceinline__ void sincos_impl(float x, float* s, float* c) { sincosf(x, s, c); }
    static __device__ __forceinline__ half2 make_real2(half x, half y) { return make_half2(x, y); }
};

// ============================================================================
// Device helper functions (templated)
// ============================================================================

template<typename T>
__device__ __forceinline__
void compute_ring_geometry_t(int ring_idx, int nside,
                              T* cos_theta, T* sin_theta,
                              T* phi_0, int* n_pixels) {
    using Traits = PrecisionTraits<T>;
    int ring_i = ring_idx + 1;

    if (ring_i < nside) {
        T i2_over_3n2 = T(ring_i * ring_i) / T(3.0 * nside * nside);
        *cos_theta = T(1.0) - i2_over_3n2;
        *sin_theta = Traits::sqrt_impl(T(1.0) - (*cos_theta) * (*cos_theta));
        *phi_0 = T(Traits::PI_VAL / (2.0 * ring_i) * 0.5);
        *n_pixels = 4 * ring_i;
    } else if (ring_i > 3 * nside) {
        int mirror_i = 4 * nside - ring_i;
        T i2_over_3n2 = T(mirror_i * mirror_i) / T(3.0 * nside * nside);
        *cos_theta = -(T(1.0) - i2_over_3n2);
        *sin_theta = Traits::sqrt_impl(T(1.0) - (*cos_theta) * (*cos_theta));
        *phi_0 = T(Traits::PI_VAL / (2.0 * mirror_i) * 0.5);
        *n_pixels = 4 * mirror_i;
    } else {
        *cos_theta = T(4.0 / 3.0) - T(2.0 * ring_i) / T(3.0 * nside);
        *sin_theta = Traits::sqrt_impl(T(1.0) - (*cos_theta) * (*cos_theta));
        int s = (ring_i % 2 == 0) ? 1 : 2;
        *phi_0 = T(Traits::PI_VAL / (2.0 * nside) * (1.0 - s / 2.0));
        *n_pixels = 4 * nside;
    }
}

// ============================================================================
// Ylm accumulation (templated)
// T = storage/output type, R = recurrence precision type
// ============================================================================

template<typename T, typename R>
__device__ void ylm_accumulate_column_v5(
    int m, int l_max,
    R cos_theta, R sin_theta,
    T gm_north_re, T gm_north_im,
    T gm_south_re, T gm_south_im,
    bool is_equator, int lp1,
    T* alm_real, T* alm_imag
) {
    using Traits = PrecisionTraits<R>;

    // Y[m,m] = (-1)^m * sin^m(theta) * sqrt((2m+1)!!/(2m)!!) / sqrt(4*pi)
    R Ymm = R(1.0) / Traits::sqrt_impl(R(4.0 * Traits::PI_VAL));

    for (int k = 1; k <= m; k++) {
        Ymm *= -sin_theta * Traits::sqrt_impl(R(2.0 * k + 1.0) / R(2.0 * k));
    }

    // Accumulate Y[m,m]
    {
        int l = m;
        int sign_ns = ((l + m) % 2 == 0) ? 1 : -1;
        T gm_re = gm_north_re + (is_equator ? T(0) : T(sign_ns) * gm_south_re);
        T gm_im = gm_north_im + (is_equator ? T(0) : T(sign_ns) * gm_south_im);
        atomicAdd(&alm_real[l * lp1 + m], T(Ymm) * gm_re);
        atomicAdd(&alm_imag[l * lp1 + m], T(Ymm) * gm_im);
    }

    if (m == l_max) return;

    // Y[m+1,m] = cos(theta) * sqrt(2m+3) * Y[m,m]
    R Ym1m = cos_theta * Traits::sqrt_impl(R(2.0 * m + 3.0)) * Ymm;

    {
        int l = m + 1;
        int sign_ns = ((l + m) % 2 == 0) ? 1 : -1;
        T gm_re = gm_north_re + (is_equator ? T(0) : T(sign_ns) * gm_south_re);
        T gm_im = gm_north_im + (is_equator ? T(0) : T(sign_ns) * gm_south_im);
        atomicAdd(&alm_real[l * lp1 + m], T(Ym1m) * gm_re);
        atomicAdd(&alm_imag[l * lp1 + m], T(Ym1m) * gm_im);
    }

    if (m + 1 == l_max) return;

    // Recurrence: Y[l,m] = A[l,m] * cos(theta) * Y[l-1,m] - B[l,m] * Y[l-2,m]
    R Ylm2 = Ymm;
    R Ylm1 = Ym1m;

    for (int l = m + 2; l <= l_max; l++) {
        R l2 = R(l * l);
        R m2 = R(m * m);
        R lm1_2 = R((l - 1) * (l - 1));

        R A_lm = Traits::sqrt_impl((R(4.0) * l2 - R(1.0)) / (l2 - m2));
        R B_lm = Traits::sqrt_impl((R(2.0 * l + 1.0)) / (R(2.0 * l - 3.0)) * (lm1_2 - m2) / (l2 - m2));

        R Ylm = A_lm * cos_theta * Ylm1 - B_lm * Ylm2;

        int sign_ns = ((l + m) % 2 == 0) ? 1 : -1;
        T gm_re = gm_north_re + (is_equator ? T(0) : T(sign_ns) * gm_south_re);
        T gm_im = gm_north_im + (is_equator ? T(0) : T(sign_ns) * gm_south_im);
        atomicAdd(&alm_real[l * lp1 + m], T(Ylm) * gm_re);
        atomicAdd(&alm_imag[l * lp1 + m], T(Ylm) * gm_im);

        Ylm2 = Ylm1;
        Ylm1 = Ylm;
    }
}

// ============================================================================
// Main fused kernel (templated)
// T = storage type, R = recurrence precision type
// ============================================================================

template<typename T, typename R>
__global__ void map2alm_fused_kernel_v5(
    int nside, int l_max, int n_maps, int n_rings,
    const T* __restrict__ map_in,
    T* __restrict__ alm_real,
    T* __restrict__ alm_imag
) {
    using Traits = PrecisionTraits<T>;
    using accum_t = typename Traits::accum_type;

    int north_ring_idx = blockIdx.x;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;
    int max_ring_pixels = 4 * nside;

    if (north_ring_idx >= n_north_rings) return;

    int south_ring_idx = n_rings - 1 - north_ring_idx;
    bool is_equator = (north_ring_idx == 2 * nside - 1);

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ char shared_mem_raw[];
    T* ring_north_batch = (T*)shared_mem_raw;
    T* ring_south_batch = ring_north_batch + PIXEL_BATCH_SIZE;

    // Compute ring geometry in recurrence precision
    R cos_theta_n, sin_theta_n;
    double phi_0_n_d;  // Phase always in double for accuracy
    int n_pixels_n;
    {
        T cos_t, sin_t, phi0_t;
        compute_ring_geometry_t<T>(north_ring_idx, nside, &cos_t, &sin_t, &phi0_t, &n_pixels_n);
        cos_theta_n = R(cos_t);
        sin_theta_n = R(sin_t);
        phi_0_n_d = double(phi0_t);
    }

    R cos_theta_s, sin_theta_s;
    double phi_0_s_d = 0.0;
    int n_pixels_s = 0;
    if (!is_equator) {
        T cos_t, sin_t, phi0_t;
        compute_ring_geometry_t<T>(south_ring_idx, nside, &cos_t, &sin_t, &phi0_t, &n_pixels_s);
        cos_theta_s = R(cos_t);
        sin_theta_s = R(sin_t);
        phi_0_s_d = double(phi0_t);
    }

    // Thread-local Gm storage (use accumulation type for precision)
    accum_t gm_north_re[MAX_M_PER_THREAD];
    accum_t gm_north_im[MAX_M_PER_THREAD];
    accum_t gm_south_re[MAX_M_PER_THREAD];
    accum_t gm_south_im[MAX_M_PER_THREAD];

    for (int t = 0; t < n_maps; t++) {
        const T* map_t = map_in + (size_t)t * n_rings * max_ring_pixels;

        // Initialize
        for (int i = 0; i < MAX_M_PER_THREAD; i++) {
            gm_north_re[i] = accum_t(0);
            gm_north_im[i] = accum_t(0);
            gm_south_re[i] = accum_t(0);
            gm_south_im[i] = accum_t(0);
        }

        // Process north ring
        for (int pix_start = 0; pix_start < n_pixels_n; pix_start += PIXEL_BATCH_SIZE) {
            int batch_size = min(PIXEL_BATCH_SIZE, n_pixels_n - pix_start);

            for (int j = tid; j < batch_size; j += block_size) {
                ring_north_batch[j] = map_t[north_ring_idx * max_ring_pixels + pix_start + j];
            }
            __syncthreads();

            int m_idx = 0;
            for (int m = tid; m <= l_max; m += block_size, m_idx++) {
                int m_eff = m % n_pixels_n;
                for (int jj = 0; jj < batch_size; jj++) {
                    int j = pix_start + jj;
                    accum_t phi_j = accum_t(2.0 * Traits::PI_VAL * m_eff * j / n_pixels_n);
                    accum_t c = cos(phi_j);
                    accum_t s = sin(phi_j);
                    accum_t val = accum_t(ring_north_batch[jj]);
                    gm_north_re[m_idx] += val * c;
                    gm_north_im[m_idx] -= val * s;
                }
            }
            __syncthreads();
        }

        // Process south ring
        if (!is_equator) {
            for (int pix_start = 0; pix_start < n_pixels_s; pix_start += PIXEL_BATCH_SIZE) {
                int batch_size = min(PIXEL_BATCH_SIZE, n_pixels_s - pix_start);

                for (int j = tid; j < batch_size; j += block_size) {
                    ring_south_batch[j] = map_t[south_ring_idx * max_ring_pixels + pix_start + j];
                }
                __syncthreads();

                int m_idx = 0;
                for (int m = tid; m <= l_max; m += block_size, m_idx++) {
                    int m_eff = m % n_pixels_s;
                    for (int jj = 0; jj < batch_size; jj++) {
                        int j = pix_start + jj;
                        accum_t phi_j = accum_t(2.0 * Traits::PI_VAL * m_eff * j / n_pixels_s);
                        accum_t c = cos(phi_j);
                        accum_t s = sin(phi_j);
                        accum_t val = accum_t(ring_south_batch[jj]);
                        gm_south_re[m_idx] += val * c;
                        gm_south_im[m_idx] -= val * s;
                    }
                }
                __syncthreads();
            }
        }

        // Apply phase corrections and accumulate
        T* alm_real_t = alm_real + (size_t)t * lp1 * lp1;
        T* alm_imag_t = alm_imag + (size_t)t * lp1 * lp1;

        int m_idx = 0;
        for (int m = tid; m <= l_max; m += block_size, m_idx++) {
            // Phase correction (always in double for accuracy)
            double phase_n = -m * phi_0_n_d;
            double cos_pn = cos(phase_n), sin_pn = sin(phase_n);
            T gm_n_re = T(gm_north_re[m_idx] * cos_pn - gm_north_im[m_idx] * sin_pn);
            T gm_n_im = T(gm_north_re[m_idx] * sin_pn + gm_north_im[m_idx] * cos_pn);

            T gm_s_re = T(0), gm_s_im = T(0);
            if (!is_equator) {
                double phase_s = -m * phi_0_s_d;
                double cos_ps = cos(phase_s), sin_ps = sin(phase_s);
                gm_s_re = T(gm_south_re[m_idx] * cos_ps - gm_south_im[m_idx] * sin_ps);
                gm_s_im = T(gm_south_re[m_idx] * sin_ps + gm_south_im[m_idx] * cos_ps);
            }

            ylm_accumulate_column_v5<T, R>(m, l_max, cos_theta_n, sin_theta_n,
                                           gm_n_re, gm_n_im, gm_s_re, gm_s_im,
                                           is_equator, lp1, alm_real_t, alm_imag_t);
        }
        __syncthreads();
    }
}

// ============================================================================
// Finalization kernel (templated)
// ============================================================================

template<typename T>
__global__ void finalize_alm_kernel_v5(
    int n_maps, int lp1, T pix_area,
    const T* __restrict__ alm_real,
    const T* __restrict__ alm_imag,
    T* __restrict__ alm_out_real,
    T* __restrict__ alm_out_imag
) {
    int t = blockIdx.x;
    int l = blockIdx.y * blockDim.x + threadIdx.x;
    int m = blockIdx.z * blockDim.y + threadIdx.y;

    if (t >= n_maps || l >= lp1 || m > l) return;

    size_t idx = (size_t)t * lp1 * lp1 + l * lp1 + m;
    alm_out_real[idx] = alm_real[idx] * pix_area;
    alm_out_imag[idx] = alm_imag[idx] * pix_area;
}

// ============================================================================
// Host wrapper functions - T = storage type, R = recurrence precision
// ============================================================================

template<typename T, typename R>
void map2alm_cuda_v5_impl(int nside, int l_max, int n_maps,
                          const T* map_in, T* alm_out_real, T* alm_out_imag) {
    int n_rings = 4 * nside - 1;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;

    int block_size = 256;
    int m_per_thread = (lp1 + block_size - 1) / block_size;
    if (m_per_thread > MAX_M_PER_THREAD) {
        fprintf(stderr, "Error: l_max=%d requires %d m values per thread, max is %d\n",
                l_max, m_per_thread, MAX_M_PER_THREAD);
        return;
    }

    // Allocate accumulation buffers
    T *alm_real, *alm_imag;
    size_t alm_size = (size_t)n_maps * lp1 * lp1 * sizeof(T);
    CUDA_CHECK(cudaMalloc(&alm_real, alm_size));
    CUDA_CHECK(cudaMalloc(&alm_imag, alm_size));
    CUDA_CHECK(cudaMemset(alm_real, 0, alm_size));
    CUDA_CHECK(cudaMemset(alm_imag, 0, alm_size));

    size_t shared_size = 2 * PIXEL_BATCH_SIZE * sizeof(T);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (shared_size > prop.sharedMemPerBlock) {
        fprintf(stderr, "Error: Required shared memory %zu exceeds limit %zu\n",
                shared_size, prop.sharedMemPerBlock);
        cudaFree(alm_real);
        cudaFree(alm_imag);
        return;
    }

    int grid_size = n_north_rings;
    map2alm_fused_kernel_v5<T, R><<<grid_size, block_size, shared_size>>>(
        nside, l_max, n_maps, n_rings, map_in, alm_real, alm_imag
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Finalize
    T pix_area = T(4.0 * M_PI / (12.0 * nside * nside));

    dim3 block_final(16, 16);
    dim3 grid_final(n_maps, CEILDIV(lp1, 16), CEILDIV(lp1, 16));
    finalize_alm_kernel_v5<T><<<grid_final, block_final>>>(
        n_maps, lp1, pix_area, alm_real, alm_imag, alm_out_real, alm_out_imag
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(alm_real);
    cudaFree(alm_imag);
}

// ============================================================================
// C API entry points
// Naming: map2alm_cuda_v5_{storage}_{recurrence}
//   storage: f64/f32 (data storage precision)
//   recurrence: f64/f32 (Ylm recurrence precision)
// ============================================================================

extern "C" {

// Float64 storage, Float64 recurrence (highest accuracy, default)
void map2alm_cuda_v5_f64_f64(int nside, int l_max, int n_maps,
                              const double* map_in, double* alm_out_real, double* alm_out_imag) {
    map2alm_cuda_v5_impl<double, double>(nside, l_max, n_maps, map_in, alm_out_real, alm_out_imag);
}

// Float64 storage, Float32 recurrence (faster recurrence, may lose accuracy at high l)
void map2alm_cuda_v5_f64_f32(int nside, int l_max, int n_maps,
                              const double* map_in, double* alm_out_real, double* alm_out_imag) {
    map2alm_cuda_v5_impl<double, float>(nside, l_max, n_maps, map_in, alm_out_real, alm_out_imag);
}

// Float32 storage, Float64 recurrence (fast I/O, accurate recurrence)
void map2alm_cuda_v5_f32_f64(int nside, int l_max, int n_maps,
                              const float* map_in, float* alm_out_real, float* alm_out_imag) {
    map2alm_cuda_v5_impl<float, double>(nside, l_max, n_maps, map_in, alm_out_real, alm_out_imag);
}

// Float32 storage, Float32 recurrence (fastest, lowest accuracy)
void map2alm_cuda_v5_f32_f32(int nside, int l_max, int n_maps,
                              const float* map_in, float* alm_out_real, float* alm_out_imag) {
    map2alm_cuda_v5_impl<float, float>(nside, l_max, n_maps, map_in, alm_out_real, alm_out_imag);
}

// Backwards-compatible aliases
void map2alm_cuda_v5_f64(int nside, int l_max, int n_maps,
                         const double* map_in, double* alm_out_real, double* alm_out_imag) {
    map2alm_cuda_v5_f64_f64(nside, l_max, n_maps, map_in, alm_out_real, alm_out_imag);
}

void map2alm_cuda_v5_f32(int nside, int l_max, int n_maps,
                         const float* map_in, float* alm_out_real, float* alm_out_imag) {
    map2alm_cuda_v5_f32_f32(nside, l_max, n_maps, map_in, alm_out_real, alm_out_imag);
}

// Wrapper that outputs complex_t for compatibility with existing API (uses f64_f64)
void map2alm_cuda_v5(int nside, int l_max, int n_maps,
                     const real_t* map_in, complex_t* alm_out) {
    int lp1 = l_max + 1;
    size_t alm_size = (size_t)n_maps * lp1 * lp1;

    // Allocate temporary buffers
    double *alm_real, *alm_imag;
    CUDA_CHECK(cudaMalloc(&alm_real, alm_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&alm_imag, alm_size * sizeof(double)));

    map2alm_cuda_v5_impl<double, double>(nside, l_max, n_maps, map_in, alm_real, alm_imag);

    // Copy to complex output (interleave real/imag)
    double* alm_out_ptr = (double*)alm_out;
    CUDA_CHECK(cudaMemcpy2D(alm_out_ptr, 2 * sizeof(double),
                            alm_real, sizeof(double),
                            sizeof(double), alm_size,
                            cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(alm_out_ptr + 1, 2 * sizeof(double),
                            alm_imag, sizeof(double),
                            sizeof(double), alm_size,
                            cudaMemcpyDeviceToDevice));

    cudaFree(alm_real);
    cudaFree(alm_imag);
}

} // extern "C"
