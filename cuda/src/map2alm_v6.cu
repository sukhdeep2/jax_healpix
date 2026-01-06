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
    R* __restrict__ Gm_even_re,   // [n_maps, lp1, n_north_rings] - optimized for coalesced reads
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
            // Layout: [n_maps, lp1, n_north_rings] for coalesced reads in Phase 2
            size_t idx = (size_t)t * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            Gm_even_re[idx] = R(gn_re + gs_re);
            Gm_even_im[idx] = R(gn_im + gs_im);
            Gm_odd_re[idx]  = R(gn_re - gs_re);
            Gm_odd_im[idx]  = R(gn_im - gs_im);
        }
    }
}

// ============================================================================
// Phase 2: Reduce to alm using warp-per-m with ring batching and multi-map
// One warp per m value, processes rings in batches to fit shared memory
// Ylm is computed once and reused across all maps in parallel
// ============================================================================

// Default ring batch size for v6 - can be reduced for multi-map
#undef RING_BATCH_SIZE
#define RING_BATCH_SIZE 256

// Maximum maps that can be processed in parallel (shared memory limited)
// Shared mem layout: geometry (2 arrays, shared) + Gm (4 arrays per map)
// f64: 2*256*8 + N*4*256*8 <= 48KB -> N <= 5
// f32: 2*256*4 + N*4*256*4 <= 48KB -> N <= 11
#define MAX_PARALLEL_MAPS_F64 5
#define MAX_PARALLEL_MAPS_F32 11

template<typename T, typename R>
__global__ void reduce_to_alm_kernel_v6(
    int nside, int l_max, int n_maps, int n_north_rings,
    int ring_batch_size, int n_maps_parallel,  // configurable parameters
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

    // Shared memory layout:
    // - Geometry (shared across maps): cos_theta, sin_theta [2 * ring_batch_size]
    // - Gm per map: Gm_even_re, Gm_even_im, Gm_odd_re, Gm_odd_im [4 * ring_batch_size each]
    extern __shared__ char smem[];
    R* sh_cos_th = (R*)smem;
    R* sh_sin_th = sh_cos_th + ring_batch_size;
    // Gm arrays for parallel maps (4 arrays per map)
    R* sh_Gm_base = sh_sin_th + ring_batch_size;

    // Per-lane Ylm recurrence state (local memory, L1 cached)
    C Ylm_prev1[MAX_RINGS_PER_LANE];
    C Ylm_prev2[MAX_RINGS_PER_LANE];

    // Per-lane accumulators for each parallel map
    // We'll process maps in groups of n_maps_parallel
    C sum_re[MAX_PARALLEL_MAPS_F32];  // Use max possible for static allocation
    C sum_im[MAX_PARALLEL_MAPS_F32];

    // Process maps in batches of n_maps_parallel
    for (int map_batch_start = 0; map_batch_start < n_maps; map_batch_start += n_maps_parallel) {
        int map_batch_end = min(map_batch_start + n_maps_parallel, n_maps);
        int n_maps_in_batch = map_batch_end - map_batch_start;

        // Process ring batches
        for (int batch_start = 0; batch_start < n_north_rings; batch_start += ring_batch_size) {
            int batch_end = min(batch_start + ring_batch_size, n_north_rings);
            int batch_size = batch_end - batch_start;

            // Cooperative load of geometry (shared across all maps)
            for (int r = lane; r < batch_size; r += 32) {
                int global_r = batch_start + r;
                sh_cos_th[r] = cos_theta[global_r];
                sh_sin_th[r] = sin_theta[global_r];
            }

            // Cooperative load of Gm for all maps in this batch
            // Gm layout: [n_maps, lp1, n_north_rings] - consecutive rings are consecutive in memory
            for (int t = 0; t < n_maps_in_batch; t++) {
                int global_t = map_batch_start + t;
                // Base index for this map and m value
                size_t base_idx = (size_t)global_t * lp1 * n_north_rings + (size_t)m * n_north_rings;
                R* sh_Gm_t = sh_Gm_base + t * 4 * ring_batch_size;  // 4 Gm arrays per map

                for (int r = lane; r < batch_size; r += 32) {
                    int global_r = batch_start + r;
                    // Now consecutive threads access consecutive memory (coalesced!)
                    size_t idx = base_idx + global_r;
                    sh_Gm_t[0 * ring_batch_size + r] = Gm_even_re[idx];
                    sh_Gm_t[1 * ring_batch_size + r] = Gm_even_im[idx];
                    sh_Gm_t[2 * ring_batch_size + r] = Gm_odd_re[idx];
                    sh_Gm_t[3 * ring_batch_size + r] = Gm_odd_im[idx];
                }
            }
            __syncwarp();

            // Determine which rings this lane handles in this batch
            int k_start = (batch_start > lane) ? (batch_start - lane + 31) / 32 : 0;
            int k_end = (batch_end > lane) ? (batch_end - 1 - lane) / 32 + 1 : 0;
            int n_my_rings_total = (n_north_rings + 31 - lane) / 32;
            k_end = min(k_end, n_my_rings_total);

            // Initialize Y[m,m] for rings in this batch
            for (int k = k_start; k < k_end; k++) {
                int global_r = lane + 32 * k;
                int local_r = global_r - batch_start;
                C sin_th = C(sh_sin_th[local_r]);

                C Ymm = C(1.0) / Traits::sqrt_d(C(4.0 * Traits::PI_VAL));
                for (int j = 1; j <= m; j++) {
                    Ymm *= -sin_th * Traits::sqrt_d(C(2*j + 1) / C(2*j));
                }

                Ylm_prev1[k] = Ymm;
                Ylm_prev2[k] = C(0);
            }

            // Process l = m to l_max for this batch
            // Ylm computed once, reused for all maps in parallel
            for (int l = m; l <= l_max; l++) {
                // Reset accumulators for all maps
                for (int t = 0; t < n_maps_in_batch; t++) {
                    sum_re[t] = C(0);
                    sum_im[t] = C(0);
                }

                // Accumulate across rings (Ylm computed once)
                for (int k = k_start; k < k_end; k++) {
                    int global_r = lane + 32 * k;
                    int local_r = global_r - batch_start;
                    C cos_th = C(sh_cos_th[local_r]);
                    C Ylm;

                    // Compute Ylm (same for all maps)
                    if (l == m) {
                        Ylm = Ylm_prev1[k];
                    } else if (l == m + 1) {
                        Ylm = cos_th * Traits::sqrt_d(C(2*m + 3)) * Ylm_prev1[k];
                        Ylm_prev2[k] = Ylm_prev1[k];
                        Ylm_prev1[k] = Ylm;
                    } else {
                        C l2 = C(l * l);
                        C m2 = C(m * m);
                        C lm1_2 = C((l-1) * (l-1));

                        C A = Traits::sqrt_d((C(4)*l2 - C(1)) / (l2 - m2));
                        C B = Traits::sqrt_d((C(2*l + 1)) / (C(2*l - 3)) * (lm1_2 - m2) / (l2 - m2));

                        Ylm = A * cos_th * Ylm_prev1[k] - B * Ylm_prev2[k];
                        Ylm_prev2[k] = Ylm_prev1[k];
                        Ylm_prev1[k] = Ylm;
                    }

                    // Accumulate Ylm * Gm for each map (reuse Ylm)
                    int parity = (l + m) & 1;
                    for (int t = 0; t < n_maps_in_batch; t++) {
                        R* sh_Gm_t = sh_Gm_base + t * 4 * ring_batch_size;
                        C gm_re, gm_im;
                        if (parity) {
                            gm_re = C(sh_Gm_t[2 * ring_batch_size + local_r]);  // odd_re
                            gm_im = C(sh_Gm_t[3 * ring_batch_size + local_r]);  // odd_im
                        } else {
                            gm_re = C(sh_Gm_t[0 * ring_batch_size + local_r]);  // even_re
                            gm_im = C(sh_Gm_t[1 * ring_batch_size + local_r]);  // even_im
                        }
                        sum_re[t] += Ylm * gm_re;
                        sum_im[t] += Ylm * gm_im;
                    }
                }

                // Warp-level reduction and output for each map
                for (int t = 0; t < n_maps_in_batch; t++) {
                    C sr = sum_re[t];
                    C si = sum_im[t];

                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        sr += __shfl_down_sync(0xffffffff, sr, offset);
                        si += __shfl_down_sync(0xffffffff, si, offset);
                    }

                    if (lane == 0) {
                        int global_t = map_batch_start + t;
                        T* alm_re_t = alm_out_re + (size_t)global_t * lp1 * lp1;
                        T* alm_im_t = alm_out_im + (size_t)global_t * lp1 * lp1;

                        if (batch_start == 0) {
                            alm_re_t[l * lp1 + m] = T(sr * C(pix_area));
                            alm_im_t[l * lp1 + m] = T(si * C(pix_area));
                        } else {
                            alm_re_t[l * lp1 + m] = T(C(alm_re_t[l * lp1 + m]) + sr * C(pix_area));
                            alm_im_t[l * lp1 + m] = T(C(alm_im_t[l * lp1 + m]) + si * C(pix_area));
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Host wrapper implementation
// ============================================================================

// Helper function to compute optimal ring batch size and parallel maps
template<typename R>
void compute_v6_params(int n_maps, int* ring_batch_size, int* n_maps_parallel) {
    // Available shared memory (48KB)
    const size_t MAX_SMEM = 48 * 1024;
    const size_t elem_size = sizeof(R);

    // Default batch size
    int batch = RING_BATCH_SIZE;  // 256

    // Calculate max parallel maps for default batch size
    // Shared mem: geometry (2 arrays) + Gm (4 arrays per map)
    // smem = 2 * batch * elem + n_par * 4 * batch * elem
    // n_par = (MAX_SMEM / elem - 2 * batch) / (4 * batch)
    int max_parallel = (MAX_SMEM / elem_size - 2 * batch) / (4 * batch);

    // Cap at compile-time maximum
    if (std::is_same<R, float>::value) {
        max_parallel = min(max_parallel, MAX_PARALLEL_MAPS_F32);
    } else {
        max_parallel = min(max_parallel, MAX_PARALLEL_MAPS_F64);
    }
    max_parallel = max(1, max_parallel);

    // If n_maps <= max_parallel, use default batch and process all maps together
    if (n_maps <= max_parallel) {
        *ring_batch_size = batch;
        *n_maps_parallel = n_maps;
        return;
    }

    // Otherwise, try to balance batch size vs parallel maps
    // Option 1: Use default batch, process maps in groups of max_parallel
    // Option 2: Reduce batch size to fit more maps in parallel

    // For now, use default batch and let the kernel handle map batching
    *ring_batch_size = batch;
    *n_maps_parallel = max_parallel;
}

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

    // Phase 1: Compute Gm and geometry
    int block_size_p1 = min(256, lp1);
    compute_gm_kernel_v6<T, R><<<n_north_rings, block_size_p1>>>(
        nside, l_max, n_maps, n_rings, map_in,
        Gm_even_re, Gm_even_im, Gm_odd_re, Gm_odd_im,
        cos_theta, sin_theta
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute optimal ring batch size and parallel maps
    int ring_batch_size, n_maps_parallel;
    compute_v6_params<R>(n_maps, &ring_batch_size, &n_maps_parallel);

    // Phase 2: Reduce to alm with ring batching and multi-map
    // Shared memory: geometry (2 arrays) + Gm per map (4 arrays each)
    size_t smem_size = (2 + 4 * n_maps_parallel) * ring_batch_size * sizeof(R);

    R pix_area = R(4.0 * M_PI / (12.0 * nside * nside));

    // One block per m, 32 threads (one warp)
    reduce_to_alm_kernel_v6<T, R><<<lp1, 32, smem_size>>>(
        nside, l_max, n_maps, n_north_rings,
        ring_batch_size, n_maps_parallel,
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
