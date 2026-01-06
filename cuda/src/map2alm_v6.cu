/**
 * map2alm_v6: Optimal warp-per-m algorithm with NO atomics
 *
 * Key optimizations over v5:
 *   1. Two-phase approach: Gm computation + warp reduction
 *   2. NO atomic adds - uses warp shuffle reduction
 *   3. Gm cached in shared memory (loaded once per m)
 *   4. Ylm recurrence state cached per lane
 *   5. Only 1 warp-level sync for shared memory load
 *
 * Supports: float64, float32, bfloat16
 *
 * Algorithm:
 *   Phase 1: For each (ring, m) compute Gm via direct DFT
 *   Phase 2: For each m, reduce across rings using warp shuffles
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../include/spht_types.h"
#include <stdio.h>
#include <type_traits>

// Maximum rings per lane (n_rings / 32)
// For nside=2048: 4096/32 = 128 rings per lane
#define MAX_RINGS_PER_LANE 256

// ============================================================================
// Type traits for multi-precision support
// ============================================================================

template<typename T>
struct V6Traits;

template<>
struct V6Traits<double> {
    using storage_t = double;
    using compute_t = double;
    using complex_storage_t = double2;
    static constexpr double PI_VAL = 3.14159265358979323846;
    static constexpr double LOG_4PI_VAL = 2.5310242469692907;

    static __device__ __forceinline__ double sqrt_d(double x) { return sqrt(x); }
    static __device__ __forceinline__ double exp_d(double x) { return exp(x); }
    static __device__ __forceinline__ double log_d(double x) { return log(x); }
    static __device__ __forceinline__ void sincos_d(double x, double* s, double* c) { sincos(x, s, c); }
    static __device__ __forceinline__ double load(const double* p) { return __ldg(p); }
    static __device__ __forceinline__ double2 load2(const double2* p) { return *p; }
};

template<>
struct V6Traits<float> {
    using storage_t = float;
    using compute_t = float;
    using complex_storage_t = float2;
    static constexpr float PI_VAL = 3.14159265f;
    static constexpr float LOG_4PI_VAL = 2.53102425f;

    static __device__ __forceinline__ float sqrt_d(float x) { return sqrtf(x); }
    static __device__ __forceinline__ float exp_d(float x) { return expf(x); }
    static __device__ __forceinline__ float log_d(float x) { return logf(x); }
    static __device__ __forceinline__ void sincos_d(float x, float* s, float* c) { sincosf(x, s, c); }
    static __device__ __forceinline__ float load(const float* p) { return __ldg(p); }
    static __device__ __forceinline__ float2 load2(const float2* p) { return *p; }
};

template<>
struct V6Traits<__nv_bfloat16> {
    using storage_t = __nv_bfloat16;
    using compute_t = float;  // Compute in float for stability
    using complex_storage_t = __nv_bfloat162;
    static constexpr float PI_VAL = 3.14159265f;
    static constexpr float LOG_4PI_VAL = 2.53102425f;

    static __device__ __forceinline__ float sqrt_d(float x) { return sqrtf(x); }
    static __device__ __forceinline__ float exp_d(float x) { return expf(x); }
    static __device__ __forceinline__ float log_d(float x) { return logf(x); }
    static __device__ __forceinline__ void sincos_d(float x, float* s, float* c) { sincosf(x, s, c); }
    static __device__ __forceinline__ float load(const __nv_bfloat16* p) { return __bfloat162float(__ldg(p)); }
    static __device__ __forceinline__ float2 load2(const __nv_bfloat162* p) {
        __nv_bfloat162 v = *p;
        return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
    }
};

// ============================================================================
// Device helper: Compute ring geometry
// ============================================================================

template<typename T>
__device__ __forceinline__ void compute_ring_geom_v6(
    int ring_idx, int nside,
    T* cos_theta, T* sin_theta, T* phi_0, int* n_pixels
) {
    using Traits = V6Traits<T>;
    using C = typename Traits::compute_t;

    int ring_i = ring_idx + 1;  // 1-indexed
    C cos_th, sin_th, phi0;
    int npix;

    if (ring_i < nside) {
        // North polar cap
        C i2_3n2 = C(ring_i * ring_i) / C(3.0 * nside * nside);
        cos_th = C(1.0) - i2_3n2;
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        phi0 = C(Traits::PI_VAL) / C(2.0 * ring_i) * C(0.5);
        npix = 4 * ring_i;
    } else if (ring_i > 3 * nside) {
        // South polar cap
        int mirror_i = 4 * nside - ring_i;
        C i2_3n2 = C(mirror_i * mirror_i) / C(3.0 * nside * nside);
        cos_th = -(C(1.0) - i2_3n2);
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        phi0 = C(Traits::PI_VAL) / C(2.0 * mirror_i) * C(0.5);
        npix = 4 * mirror_i;
    } else {
        // Equatorial belt
        cos_th = C(4.0 / 3.0) - C(2.0 * ring_i) / C(3.0 * nside);
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        int s = (ring_i % 2 == 0) ? 1 : 2;
        phi0 = C(Traits::PI_VAL) / C(2.0 * nside) * C(1.0 - s / 2.0);
        npix = 4 * nside;
    }

    *cos_theta = T(cos_th);
    *sin_theta = T(sin_th);
    *phi_0 = T(phi0);
    *n_pixels = npix;
}

// ============================================================================
// Phase 1: Compute Gm_even and Gm_odd for all (ring_pair, m)
// Each block handles one north ring pair, threads handle m values
// ============================================================================

template<typename T, typename R>
__global__ void compute_gm_kernel_v6(
    int nside, int l_max, int n_maps, int n_rings,
    const T* __restrict__ map_in,
    R* __restrict__ Gm_even_re,   // [n_maps, n_north_rings, lp1]
    R* __restrict__ Gm_even_im,
    R* __restrict__ Gm_odd_re,
    R* __restrict__ Gm_odd_im,
    R* __restrict__ cos_theta_out,  // [n_north_rings]
    R* __restrict__ sin_theta_out
) {
    using Traits = V6Traits<T>;
    using C = typename Traits::compute_t;

    int north_ring = blockIdx.x;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;
    int max_pix = 4 * nside;

    if (north_ring >= n_north_rings) return;

    int south_ring = n_rings - 1 - north_ring;
    bool is_equator = (north_ring == 2 * nside - 1);

    // Compute geometry
    T cos_th_n, sin_th_n, phi0_n;
    int n_pix_n;
    compute_ring_geom_v6<T>(north_ring, nside, &cos_th_n, &sin_th_n, &phi0_n, &n_pix_n);

    T cos_th_s, sin_th_s, phi0_s;
    int n_pix_s = 0;
    if (!is_equator) {
        compute_ring_geom_v6<T>(south_ring, nside, &cos_th_s, &sin_th_s, &phi0_s, &n_pix_s);
    }

    // Store geometry (thread 0 only)
    if (threadIdx.x == 0) {
        cos_theta_out[north_ring] = R(cos_th_n);
        sin_theta_out[north_ring] = R(sin_th_n);
    }

    // Process each map
    for (int t = 0; t < n_maps; t++) {
        const T* map_t = map_in + (size_t)t * n_rings * max_pix;

        // Each thread handles one or more m values
        for (int m = threadIdx.x; m <= l_max; m += blockDim.x) {
            C gn_re = C(0), gn_im = C(0);
            C gs_re = C(0), gs_im = C(0);

            // North ring DFT with proper phase handling
            // G_m = exp(-i*m*phi_0) * sum_j f(phi_j) * exp(-i*m_eff*j*2π/n_pix)
            // where m_eff = m % n_pix for the periodic DFT part
            C dphi_n = C(2.0 * Traits::PI_VAL) / C(n_pix_n);
            int m_eff_n = m % n_pix_n;

            for (int j = 0; j < n_pix_n; j++) {
                C val = C(Traits::load(&map_t[north_ring * max_pix + j]));
                // Phase = -m*phi_0 - m_eff*j*dphi (phi_0 uses m, not m_eff!)
                C angle = -C(m) * C(phi0_n) - C(m_eff_n) * C(j) * dphi_n;
                C c, s;
                Traits::sincos_d(angle, &s, &c);
                gn_re += val * c;
                gn_im += val * s;
            }

            // South ring DFT
            if (!is_equator && n_pix_s > 0) {
                C dphi_s = C(2.0 * Traits::PI_VAL) / C(n_pix_s);
                int m_eff_s = m % n_pix_s;

                for (int j = 0; j < n_pix_s; j++) {
                    C val = C(Traits::load(&map_t[south_ring * max_pix + j]));
                    C angle = -C(m) * C(phi0_s) - C(m_eff_s) * C(j) * dphi_s;
                    C c, s;
                    Traits::sincos_d(angle, &s, &c);
                    gs_re += val * c;
                    gs_im += val * s;
                }
            }

            // Combine with N-S symmetry
            // Gm_even = Gm_n + Gm_s (for even l+m)
            // Gm_odd  = Gm_n - Gm_s (for odd l+m)
            size_t idx = (size_t)t * n_north_rings * lp1 + north_ring * lp1 + m;
            Gm_even_re[idx] = R(gn_re + gs_re);
            Gm_even_im[idx] = R(gn_im + gs_im);
            Gm_odd_re[idx]  = R(gn_re - gs_re);
            Gm_odd_im[idx]  = R(gn_im - gs_im);
        }
    }
}

// ============================================================================
// Phase 2: Reduce to alm using warp-per-m with cached Gm
// One warp per m value, processes all rings cooperatively
// ============================================================================

template<typename T, typename R>
__global__ void reduce_to_alm_kernel_v6(
    int l_max, int n_maps, int n_north_rings,
    const R* __restrict__ Gm_even_re,
    const R* __restrict__ Gm_even_im,
    const R* __restrict__ Gm_odd_re,
    const R* __restrict__ Gm_odd_im,
    const R* __restrict__ cos_theta,
    const R* __restrict__ sin_theta,
    R pix_area,
    T* __restrict__ alm_out_re,
    T* __restrict__ alm_out_im
) {
    using Traits = V6Traits<R>;
    using C = typename Traits::compute_t;

    // One block per m, use only first warp (32 threads)
    int m = blockIdx.x;
    int lane = threadIdx.x;
    int lp1 = l_max + 1;

    if (m > l_max || lane >= 32) return;

    // Shared memory for caching Gm and geometry
    extern __shared__ char smem[];
    R* sh_Gm_even_re = (R*)smem;
    R* sh_Gm_even_im = sh_Gm_even_re + n_north_rings;
    R* sh_Gm_odd_re  = sh_Gm_even_im + n_north_rings;
    R* sh_Gm_odd_im  = sh_Gm_odd_re + n_north_rings;
    R* sh_cos_th     = sh_Gm_odd_im + n_north_rings;
    R* sh_sin_th     = sh_cos_th + n_north_rings;

    // Per-lane Ylm recurrence state (local memory, L1 cached)
    int n_my_rings = (n_north_rings + 31 - lane) / 32;
    C Ylm_prev1[MAX_RINGS_PER_LANE];
    C Ylm_prev2[MAX_RINGS_PER_LANE];

    // Process each map
    for (int t = 0; t < n_maps; t++) {
        // Cooperative load of Gm and geometry into shared memory
        size_t base_idx = (size_t)t * n_north_rings * lp1;
        for (int r = lane; r < n_north_rings; r += 32) {
            size_t idx = base_idx + r * lp1 + m;
            sh_Gm_even_re[r] = Gm_even_re[idx];
            sh_Gm_even_im[r] = Gm_even_im[idx];
            sh_Gm_odd_re[r]  = Gm_odd_re[idx];
            sh_Gm_odd_im[r]  = Gm_odd_im[idx];
            if (t == 0) {  // Only load geometry once
                sh_cos_th[r] = cos_theta[r];
                sh_sin_th[r] = sin_theta[r];
            }
        }
        __syncwarp();  // Only warp-level sync needed!

        // Initialize Y[m,m] for all my rings using direct multiplication (matches v5)
        int idx = 0;
        for (int r = lane; r < n_north_rings; r += 32, idx++) {
            C sin_th = C(sh_sin_th[r]);

            // Y[m,m] = (-1)^m * sin^m(θ) * sqrt((2m+1)!!/(2m)!!) / sqrt(4π)
            // Use direct multiplication like v5 for numerical stability
            C Ymm = C(1.0) / Traits::sqrt_d(C(4.0 * Traits::PI_VAL));
            for (int k = 1; k <= m; k++) {
                Ymm *= -sin_th * Traits::sqrt_d(C(2*k + 1) / C(2*k));
            }

            Ylm_prev1[idx] = Ymm;
            Ylm_prev2[idx] = C(0);
        }

        // Output pointer for this map
        T* alm_re_t = alm_out_re + (size_t)t * lp1 * lp1;
        T* alm_im_t = alm_out_im + (size_t)t * lp1 * lp1;

        // Process l = m to l_max
        for (int l = m; l <= l_max; l++) {
            C sum_re = C(0), sum_im = C(0);

            idx = 0;
            for (int r = lane; r < n_north_rings; r += 32, idx++) {
                C cos_th = C(sh_cos_th[r]);
                C Ylm;

                if (l == m) {
                    Ylm = Ylm_prev1[idx];
                } else if (l == m + 1) {
                    // Y[m+1,m] = cos(θ) * sqrt(2m+3) * Y[m,m]
                    Ylm = cos_th * Traits::sqrt_d(C(2*m + 3)) * Ylm_prev1[idx];
                    Ylm_prev2[idx] = Ylm_prev1[idx];
                    Ylm_prev1[idx] = Ylm;
                } else {
                    // Recurrence: Y[l,m] = A*cos(θ)*Y[l-1,m] - B*Y[l-2,m]
                    C l2 = C(l * l);
                    C m2 = C(m * m);
                    C lm1_2 = C((l-1) * (l-1));

                    C A = Traits::sqrt_d((C(4)*l2 - C(1)) / (l2 - m2));
                    C B = Traits::sqrt_d((C(2*l + 1)) / (C(2*l - 3)) * (lm1_2 - m2) / (l2 - m2));

                    Ylm = A * cos_th * Ylm_prev1[idx] - B * Ylm_prev2[idx];
                    Ylm_prev2[idx] = Ylm_prev1[idx];
                    Ylm_prev1[idx] = Ylm;
                }

                // Select Gm based on (l+m) parity
                C gm_re, gm_im;
                if ((l + m) & 1) {
                    gm_re = C(sh_Gm_odd_re[r]);
                    gm_im = C(sh_Gm_odd_im[r]);
                } else {
                    gm_re = C(sh_Gm_even_re[r]);
                    gm_im = C(sh_Gm_even_im[r]);
                }

                sum_re += Ylm * gm_re;
                sum_im += Ylm * gm_im;
            }

            // Warp-level reduction (NO __syncthreads!)
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_re += __shfl_down_sync(0xffffffff, sum_re, offset);
                sum_im += __shfl_down_sync(0xffffffff, sum_im, offset);
            }

            // Lane 0 writes final result
            if (lane == 0) {
                alm_re_t[l * lp1 + m] = T(sum_re * C(pix_area));
                alm_im_t[l * lp1 + m] = T(sum_im * C(pix_area));
            }
        }
    }
}

// ============================================================================
// Host wrapper implementation
// ============================================================================

template<typename T, typename R>
void map2alm_cuda_v6_impl(
    int nside, int l_max, int n_maps,
    const T* map_in,
    T* alm_out_re, T* alm_out_im
) {
    int n_rings = 4 * nside - 1;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;

    // Check limits
    int rings_per_lane = (n_north_rings + 31) / 32;
    if (rings_per_lane > MAX_RINGS_PER_LANE) {
        fprintf(stderr, "Error: nside=%d requires %d rings per lane, max is %d\n",
                nside, rings_per_lane, MAX_RINGS_PER_LANE);
        return;
    }

    // Allocate intermediate buffers
    size_t gm_size = (size_t)n_maps * n_north_rings * lp1 * sizeof(R);
    size_t geom_size = n_north_rings * sizeof(R);

    R *Gm_even_re, *Gm_even_im, *Gm_odd_re, *Gm_odd_im;
    R *cos_theta, *sin_theta;

    CUDA_CHECK(cudaMalloc(&Gm_even_re, gm_size));
    CUDA_CHECK(cudaMalloc(&Gm_even_im, gm_size));
    CUDA_CHECK(cudaMalloc(&Gm_odd_re, gm_size));
    CUDA_CHECK(cudaMalloc(&Gm_odd_im, gm_size));
    CUDA_CHECK(cudaMalloc(&cos_theta, geom_size));
    CUDA_CHECK(cudaMalloc(&sin_theta, geom_size));

    // Phase 1: Compute Gm
    int block_size_p1 = min(256, lp1);
    compute_gm_kernel_v6<T, R><<<n_north_rings, block_size_p1>>>(
        nside, l_max, n_maps, n_rings, map_in,
        Gm_even_re, Gm_even_im, Gm_odd_re, Gm_odd_im,
        cos_theta, sin_theta
    );
    CUDA_CHECK(cudaGetLastError());

    // Phase 2: Reduce to alm
    // Shared memory: 6 arrays of n_north_rings elements
    size_t smem_size = 6 * n_north_rings * sizeof(R);

    // Check shared memory limit
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (smem_size > prop.sharedMemPerBlock) {
        fprintf(stderr, "Warning: Required shared memory %zu exceeds limit %zu\n",
                smem_size, prop.sharedMemPerBlock);
        fprintf(stderr, "Consider using smaller nside or a different approach\n");
        // Could fall back to v5 here, but for now just warn
    }

    R pix_area = R(4.0 * M_PI / (12.0 * nside * nside));

    // One block per m, 32 threads (one warp)
    reduce_to_alm_kernel_v6<T, R><<<lp1, 32, smem_size>>>(
        l_max, n_maps, n_north_rings,
        Gm_even_re, Gm_even_im, Gm_odd_re, Gm_odd_im,
        cos_theta, sin_theta, pix_area,
        alm_out_re, alm_out_im
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    cudaFree(Gm_even_re);
    cudaFree(Gm_even_im);
    cudaFree(Gm_odd_re);
    cudaFree(Gm_odd_im);
    cudaFree(cos_theta);
    cudaFree(sin_theta);
}

// ============================================================================
// C API entry points
// ============================================================================

extern "C" {

// Float64 storage, Float64 recurrence
void map2alm_cuda_v6_f64_f64(int nside, int l_max, int n_maps,
                              const double* map_in,
                              double* alm_out_re, double* alm_out_im) {
    map2alm_cuda_v6_impl<double, double>(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

// Float64 storage, Float32 recurrence
void map2alm_cuda_v6_f64_f32(int nside, int l_max, int n_maps,
                              const double* map_in,
                              double* alm_out_re, double* alm_out_im) {
    map2alm_cuda_v6_impl<double, float>(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

// Float32 storage, Float64 recurrence
void map2alm_cuda_v6_f32_f64(int nside, int l_max, int n_maps,
                              const float* map_in,
                              float* alm_out_re, float* alm_out_im) {
    map2alm_cuda_v6_impl<float, double>(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

// Float32 storage, Float32 recurrence
void map2alm_cuda_v6_f32_f32(int nside, int l_max, int n_maps,
                              const float* map_in,
                              float* alm_out_re, float* alm_out_im) {
    map2alm_cuda_v6_impl<float, float>(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

// BFloat16 storage, Float32 recurrence
void map2alm_cuda_v6_bf16_f32(int nside, int l_max, int n_maps,
                               const __nv_bfloat16* map_in,
                               __nv_bfloat16* alm_out_re, __nv_bfloat16* alm_out_im) {
    map2alm_cuda_v6_impl<__nv_bfloat16, float>(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

// Convenience aliases
void map2alm_cuda_v6_f64(int nside, int l_max, int n_maps,
                          const double* map_in,
                          double* alm_out_re, double* alm_out_im) {
    map2alm_cuda_v6_f64_f64(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

void map2alm_cuda_v6_f32(int nside, int l_max, int n_maps,
                          const float* map_in,
                          float* alm_out_re, float* alm_out_im) {
    map2alm_cuda_v6_f32_f32(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

// Legacy complex_t output wrapper
void map2alm_cuda_v6(int nside, int l_max, int n_maps,
                      const real_t* map_in, complex_t* alm_out) {
    int lp1 = l_max + 1;
    size_t alm_size = (size_t)n_maps * lp1 * lp1;

    // Allocate temporary separate real/imag buffers
    double *alm_re, *alm_im;
    CUDA_CHECK(cudaMalloc(&alm_re, alm_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&alm_im, alm_size * sizeof(double)));

    map2alm_cuda_v6_f64_f64(nside, l_max, n_maps, map_in, alm_re, alm_im);

    // Interleave to complex output
    double* alm_out_ptr = (double*)alm_out;
    CUDA_CHECK(cudaMemcpy2D(alm_out_ptr, 2 * sizeof(double),
                            alm_re, sizeof(double),
                            sizeof(double), alm_size,
                            cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(alm_out_ptr + 1, 2 * sizeof(double),
                            alm_im, sizeof(double),
                            sizeof(double), alm_size,
                            cudaMemcpyDeviceToDevice));

    cudaFree(alm_re);
    cudaFree(alm_im);
}

} // extern "C"
