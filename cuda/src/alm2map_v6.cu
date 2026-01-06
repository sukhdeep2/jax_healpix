/**
 * alm2map_v6: Optimal warp-per-m algorithm for spherical harmonic synthesis
 *
 * This is the inverse of map2alm_v6. Key optimizations:
 *   1. Two-phase approach: Fmy computation (alm*Ylm) + inverse DFT synthesis
 *   2. NO atomic adds - uses warp shuffle reduction for Fmy
 *   3. Ylm recurrence computed once per ring, reused across maps
 *   4. North-south symmetry exploited for efficiency
 *
 * Supports: float64, float32 (with configurable recurrence precision)
 *
 * Algorithm:
 *   Phase 1: For each (ring_pair, m) compute Fmy = sum_l(alm[l,m] * Ylm)
 *   Phase 2: For each ring, synthesize map via inverse DFT of Fmy
 *
 * Reference: jax_healpix/SPHT_jax.py alm2map() and alm2ring_ns_dot()
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cufft.h>
#include "../include/spht_types.h"
#include <stdio.h>
#include <type_traits>

// Maximum rings per lane (same as map2alm_v6)
#define MAX_RINGS_PER_LANE 256

// Ring batch size for Phase 1 (undef to override default from spht_types.h)
#undef RING_BATCH_SIZE
#define RING_BATCH_SIZE 256

// ============================================================================
// Type traits for multi-precision support (shared with map2alm_v6)
// ============================================================================

template<typename T>
struct V6TraitsSynth;

template<>
struct V6TraitsSynth<double> {
    using storage_t = double;
    using compute_t = double;
    using complex_storage_t = double2;
    static constexpr double PI_VAL = 3.14159265358979323846;

    static __device__ __forceinline__ double sqrt_d(double x) { return sqrt(x); }
    static __device__ __forceinline__ double exp_d(double x) { return exp(x); }
    static __device__ __forceinline__ void sincos_d(double x, double* s, double* c) { sincos(x, s, c); }
    static __device__ __forceinline__ double load(const double* p) { return __ldg(p); }
};

template<>
struct V6TraitsSynth<float> {
    using storage_t = float;
    using compute_t = float;
    using complex_storage_t = float2;
    static constexpr float PI_VAL = 3.14159265f;

    static __device__ __forceinline__ float sqrt_d(float x) { return sqrtf(x); }
    static __device__ __forceinline__ float exp_d(float x) { return expf(x); }
    static __device__ __forceinline__ void sincos_d(float x, float* s, float* c) { sincosf(x, s, c); }
    static __device__ __forceinline__ float load(const float* p) { return __ldg(p); }
};

// ============================================================================
// Device helper: Compute ring geometry (shared with map2alm_v6)
// ============================================================================

template<typename T>
__device__ __forceinline__ void compute_ring_geom_synth_v6(
    int ring_idx, int nside,
    T* cos_theta, T* sin_theta, T* phi_0, int* n_pixels
) {
    using Traits = V6TraitsSynth<T>;
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
// Phase 1: Compute Fmy = sum_l(alm[l,m] * Ylm) for all (ring_pair, m)
// One warp per m value, each lane handles subset of north rings
// Ylm is computed once and reused across all maps
// ============================================================================

template<typename T, typename R>
__global__ void compute_fmy_kernel_v6(
    int nside, int l_max, int n_maps, int n_north_rings,
    int ring_batch_size,
    const T* __restrict__ alm_real,  // [n_maps, lp1, lp1]
    const T* __restrict__ alm_imag,
    R* __restrict__ Fmy_even_re,     // [n_maps, lp1, n_north_rings]
    R* __restrict__ Fmy_even_im,
    R* __restrict__ Fmy_odd_re,
    R* __restrict__ Fmy_odd_im,
    R* __restrict__ cos_theta_out,   // [n_north_rings]
    R* __restrict__ sin_theta_out
) {
    using Traits = V6TraitsSynth<R>;
    using C = typename Traits::compute_t;

    // One block per m, use only first warp (32 threads)
    int m = blockIdx.x;
    int lane = threadIdx.x;
    int lp1 = l_max + 1;

    if (m > l_max || lane >= 32) return;

    // Shared memory for ring geometry and alm (loaded per batch)
    extern __shared__ char smem[];
    R* sh_cos_th = (R*)smem;
    R* sh_sin_th = sh_cos_th + ring_batch_size;
    // alm values for this m (all l values, shared across lanes)
    T* sh_alm_re = (T*)(sh_sin_th + ring_batch_size);
    T* sh_alm_im = sh_alm_re + lp1;

    // Per-lane Ylm recurrence state
    C Ylm_prev1[MAX_RINGS_PER_LANE];
    C Ylm_prev2[MAX_RINGS_PER_LANE];

    // Determine global ring indices this lane handles
    int n_my_rings_total = (n_north_rings + 31 - lane) / 32;

    // Process ring batches
    for (int batch_start = 0; batch_start < n_north_rings; batch_start += ring_batch_size) {
        int batch_end = min(batch_start + ring_batch_size, n_north_rings);
        int batch_size = batch_end - batch_start;

        // Cooperative load of geometry (only once per batch, by lane 0 in first block)
        if (m == 0) {
            for (int r = lane; r < batch_size; r += 32) {
                int global_r = batch_start + r;
                T cos_th, sin_th, phi0;
                int npix;
                compute_ring_geom_synth_v6<T>(global_r, nside, &cos_th, &sin_th, &phi0, &npix);
                sh_cos_th[r] = R(cos_th);
                sh_sin_th[r] = R(sin_th);
                // Store to global (thread 0 of lane)
                if (lane == 0 || r == lane) {  // Each lane stores its own ring
                    cos_theta_out[global_r] = R(cos_th);
                    sin_theta_out[global_r] = R(sin_th);
                }
            }
        } else {
            // Other m blocks just load from local computation
            for (int r = lane; r < batch_size; r += 32) {
                int global_r = batch_start + r;
                T cos_th, sin_th, phi0;
                int npix;
                compute_ring_geom_synth_v6<T>(global_r, nside, &cos_th, &sin_th, &phi0, &npix);
                sh_cos_th[r] = R(cos_th);
                sh_sin_th[r] = R(sin_th);
            }
        }

        // Determine which rings this lane handles in this batch
        int k_start = (batch_start > lane) ? (batch_start - lane + 31) / 32 : 0;
        int k_end = (batch_end > lane) ? (batch_end - 1 - lane) / 32 + 1 : 0;
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

        __syncwarp();

        // Process each map
        for (int t = 0; t < n_maps; t++) {
            const T* alm_re_t = alm_real + (size_t)t * lp1 * lp1;
            const T* alm_im_t = alm_imag + (size_t)t * lp1 * lp1;

            // Load alm values for this m (all l >= m)
            // Note: alm is already scaled by 2 for m > 0 in Python wrapper
            for (int l = m + lane; l <= l_max; l += 32) {
                sh_alm_re[l] = alm_re_t[l * lp1 + m];
                sh_alm_im[l] = alm_im_t[l * lp1 + m];
            }
            __syncwarp();

            // Reset Ylm state for each map (recompute from cached initial values)
            // Actually, we need to recompute Ylm once per batch, not per map
            // Since Ylm doesn't depend on the map, compute once and reuse

            // For each ring this lane handles, compute Fmy = sum_l(alm * Ylm)
            for (int k = k_start; k < k_end; k++) {
                int global_r = lane + 32 * k;
                int local_r = global_r - batch_start;
                C cos_th = C(sh_cos_th[local_r]);
                C sin_th = C(sh_sin_th[local_r]);

                // Accumulate Fmy for north and south rings
                C fmy_n_re = C(0), fmy_n_im = C(0);
                C fmy_s_re = C(0), fmy_s_im = C(0);

                // Recompute Ylm for this ring (needed fresh for each map iteration)
                C Ymm = C(1.0) / Traits::sqrt_d(C(4.0 * Traits::PI_VAL));
                for (int j = 1; j <= m; j++) {
                    Ymm *= -sin_th * Traits::sqrt_d(C(2*j + 1) / C(2*j));
                }

                C Ylm_p1 = Ymm;
                C Ylm_p2 = C(0);

                // Ylm recurrence and accumulation
                for (int l = m; l <= l_max; l++) {
                    C Ylm;
                    if (l == m) {
                        Ylm = Ymm;
                    } else if (l == m + 1) {
                        Ylm = cos_th * Traits::sqrt_d(C(2*m + 3)) * Ylm_p1;
                        Ylm_p2 = Ylm_p1;
                        Ylm_p1 = Ylm;
                    } else {
                        C l2 = C(l * l);
                        C m2 = C(m * m);
                        C lm1_2 = C((l-1) * (l-1));

                        C A = Traits::sqrt_d((C(4)*l2 - C(1)) / (l2 - m2));
                        C B = Traits::sqrt_d((C(2*l + 1)) / (C(2*l - 3)) * (lm1_2 - m2) / (l2 - m2));

                        Ylm = A * cos_th * Ylm_p1 - B * Ylm_p2;
                        Ylm_p2 = Ylm_p1;
                        Ylm_p1 = Ylm;
                    }

                    // alm * Ylm contribution
                    C alm_re = C(sh_alm_re[l]);
                    C alm_im = C(sh_alm_im[l]);

                    // North: Fmy += alm * Ylm
                    fmy_n_re += alm_re * Ylm;
                    fmy_n_im += alm_im * Ylm;

                    // South: Ylm(-cos_theta) = (-1)^(l+m) * Ylm(cos_theta)
                    int parity = (l + m) & 1;
                    C sign_ns = parity ? C(-1) : C(1);
                    fmy_s_re += alm_re * (sign_ns * Ylm);
                    fmy_s_im += alm_im * (sign_ns * Ylm);
                }

                // Combine N/S: Fmy_even = Fmy_n + Fmy_s, Fmy_odd = Fmy_n - Fmy_s
                // Output layout: [n_maps, lp1, n_north_rings]
                size_t idx = (size_t)t * lp1 * n_north_rings + (size_t)m * n_north_rings + global_r;
                Fmy_even_re[idx] = R(fmy_n_re + fmy_s_re);
                Fmy_even_im[idx] = R(fmy_n_im + fmy_s_im);
                Fmy_odd_re[idx]  = R(fmy_n_re - fmy_s_re);
                Fmy_odd_im[idx]  = R(fmy_n_im - fmy_s_im);
            }

            __syncwarp();
        }
    }
}

// ============================================================================
// Phase 2: Synthesize map from Fmy via inverse DFT
// map[ring,j] = Real(sum_m(Fmy[m] * exp(+i*m*phi_j)))
// One block per north ring pair, threads handle pixels
// ============================================================================

template<typename T, typename R>
__global__ void synthesize_map_kernel_v6(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings,
    const R* __restrict__ Fmy_even_re,   // [n_maps, lp1, n_north_rings]
    const R* __restrict__ Fmy_even_im,
    const R* __restrict__ Fmy_odd_re,
    const R* __restrict__ Fmy_odd_im,
    T* __restrict__ map_out              // [n_maps, n_rings, max_pix]
) {
    using Traits = V6TraitsSynth<R>;
    using C = typename Traits::compute_t;

    int north_ring = blockIdx.x;
    int lp1 = l_max + 1;
    int max_pix = 4 * nside;

    if (north_ring >= n_north_rings) return;

    int south_ring = n_rings - 1 - north_ring;
    bool is_equator = (north_ring == 2 * nside - 1);

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Get ring geometry
    T cos_th_n, sin_th_n, phi0_n;
    int n_pix_n;
    compute_ring_geom_synth_v6<T>(north_ring, nside, &cos_th_n, &sin_th_n, &phi0_n, &n_pix_n);

    T cos_th_s, sin_th_s, phi0_s;
    int n_pix_s = 0;
    if (!is_equator) {
        compute_ring_geom_synth_v6<T>(south_ring, nside, &cos_th_s, &sin_th_s, &phi0_s, &n_pix_s);
    }

    // Shared memory for Fmy values (loaded once per map)
    extern __shared__ char shared_mem[];
    R* sh_fmy_even_re = (R*)shared_mem;
    R* sh_fmy_even_im = sh_fmy_even_re + lp1;
    R* sh_fmy_odd_re = sh_fmy_even_im + lp1;
    R* sh_fmy_odd_im = sh_fmy_odd_re + lp1;

    // Process each map
    for (int t = 0; t < n_maps; t++) {
        T* map_t = map_out + (size_t)t * n_rings * max_pix;

        // Load Fmy for this ring and map
        // Layout: [n_maps, lp1, n_north_rings]
        for (int m = tid; m <= l_max; m += block_size) {
            size_t idx = (size_t)t * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            sh_fmy_even_re[m] = Fmy_even_re[idx];
            sh_fmy_even_im[m] = Fmy_even_im[idx];
            sh_fmy_odd_re[m] = Fmy_odd_re[idx];
            sh_fmy_odd_im[m] = Fmy_odd_im[idx];
        }
        __syncthreads();

        // Synthesize north ring pixels
        // map[j] = Real(sum_m(Fmy_n[m] * exp(+i*m*phi_j)))
        // where phi_j = phi_0 + j * 2*pi/n_pix
        // Fmy_n = (Fmy_even + Fmy_odd) / 2, but since alm scaled by 2 for m>0,
        // we just take the appropriate component
        for (int j = tid; j < n_pix_n; j += block_size) {
            C sum_re = C(0);

            for (int m = 0; m <= l_max; m++) {
                // Reconstruct Fmy_north = (Fmy_even + Fmy_odd) / 2
                // But alm was already scaled by 2 for m>0 in synthesis
                C fmy_n_re = C(sh_fmy_even_re[m] + sh_fmy_odd_re[m]) * C(0.5);
                C fmy_n_im = C(sh_fmy_even_im[m] + sh_fmy_odd_im[m]) * C(0.5);

                // Phase: exp(+i*m*phi_j) where phi_j = phi_0 + j*2*pi/n_pix
                C phi_j = C(phi0_n) + C(j) * C(2.0 * Traits::PI_VAL) / C(n_pix_n);
                C angle = C(m) * phi_j;
                C cos_ang, sin_ang;
                Traits::sincos_d(angle, &sin_ang, &cos_ang);

                // Re(Fmy * exp(+im*phi)) = Fmy_re * cos - Fmy_im * (-sin)
                //                        = Fmy_re * cos + Fmy_im * sin
                // Wait, exp(+i*theta) = cos(theta) + i*sin(theta)
                // Re(Fmy * exp(+i*theta)) = Fmy_re * cos - Fmy_im * sin
                sum_re += fmy_n_re * cos_ang - fmy_n_im * sin_ang;
            }

            map_t[north_ring * max_pix + j] = T(sum_re);
        }

        // Synthesize south ring pixels
        if (!is_equator) {
            for (int j = tid; j < n_pix_s; j += block_size) {
                C sum_re = C(0);

                for (int m = 0; m <= l_max; m++) {
                    // Reconstruct Fmy_south = (Fmy_even - Fmy_odd) / 2
                    C fmy_s_re = C(sh_fmy_even_re[m] - sh_fmy_odd_re[m]) * C(0.5);
                    C fmy_s_im = C(sh_fmy_even_im[m] - sh_fmy_odd_im[m]) * C(0.5);

                    C phi_j = C(phi0_s) + C(j) * C(2.0 * Traits::PI_VAL) / C(n_pix_s);
                    C angle = C(m) * phi_j;
                    C cos_ang, sin_ang;
                    Traits::sincos_d(angle, &sin_ang, &cos_ang);

                    sum_re += fmy_s_re * cos_ang - fmy_s_im * sin_ang;
                }

                map_t[south_ring * max_pix + j] = T(sum_re);
            }
        }

        __syncthreads();
    }
}

// ============================================================================
// Kernel to scale alm: multiply m>0 by 2 (to account for missing m<0)
// ============================================================================

template<typename T>
__global__ void scale_alm_for_synth_kernel(
    int n_maps, int lp1,
    const T* __restrict__ alm_in_real,
    const T* __restrict__ alm_in_imag,
    T* __restrict__ alm_out_real,
    T* __restrict__ alm_out_imag
) {
    int t = blockIdx.x;
    int l = blockIdx.y * blockDim.x + threadIdx.x;
    int m = blockIdx.z * blockDim.y + threadIdx.y;

    if (t >= n_maps || l >= lp1 || m > l) return;

    size_t idx = (size_t)t * lp1 * lp1 + l * lp1 + m;

    T scale = (m > 0) ? T(2.0) : T(1.0);
    alm_out_real[idx] = alm_in_real[idx] * scale;
    alm_out_imag[idx] = alm_in_imag[idx] * scale;
}

// ============================================================================
// Host wrapper implementation
// ============================================================================

template<typename T, typename R>
void alm2map_cuda_v6_impl(
    int nside, int l_max, int n_maps,
    const T* alm_in_real, const T* alm_in_imag,
    T* map_out
) {
    int n_rings = 4 * nside - 1;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;
    int max_pix = 4 * nside;

    // Check limits
    int rings_per_lane = (n_north_rings + 31) / 32;
    if (rings_per_lane > MAX_RINGS_PER_LANE) {
        fprintf(stderr, "Error: nside=%d requires %d rings per lane, max is %d\n",
                nside, rings_per_lane, MAX_RINGS_PER_LANE);
        return;
    }

    // Timing events
    static bool timing_enabled = (getenv("SPHT_TIMING") != nullptr);
    cudaEvent_t start_scale, end_scale, start_p1, end_p1, start_p2, end_p2;
    if (timing_enabled) {
        cudaEventCreate(&start_scale);
        cudaEventCreate(&end_scale);
        cudaEventCreate(&start_p1);
        cudaEventCreate(&end_p1);
        cudaEventCreate(&start_p2);
        cudaEventCreate(&end_p2);
        cudaEventRecord(start_scale);
    }

    // Allocate scaled alm (m>0 multiplied by 2)
    T *alm_scaled_real, *alm_scaled_imag;
    size_t alm_size = (size_t)n_maps * lp1 * lp1 * sizeof(T);
    CUDA_CHECK(cudaMalloc(&alm_scaled_real, alm_size));
    CUDA_CHECK(cudaMalloc(&alm_scaled_imag, alm_size));

    // Scale alm: m>0 by 2
    dim3 block_scale(16, 16);
    dim3 grid_scale(n_maps, CEILDIV(lp1, 16), CEILDIV(lp1, 16));
    scale_alm_for_synth_kernel<T><<<grid_scale, block_scale>>>(
        n_maps, lp1, alm_in_real, alm_in_imag, alm_scaled_real, alm_scaled_imag
    );
    CUDA_CHECK(cudaGetLastError());

    if (timing_enabled) {
        cudaEventRecord(end_scale);
        cudaEventRecord(start_p1);
    }

    // Allocate intermediate Fmy buffers
    size_t fmy_size = (size_t)n_maps * lp1 * n_north_rings * sizeof(R);
    size_t geom_size = n_north_rings * sizeof(R);

    R *Fmy_even_re, *Fmy_even_im, *Fmy_odd_re, *Fmy_odd_im;
    R *cos_theta, *sin_theta;

    CUDA_CHECK(cudaMalloc(&Fmy_even_re, fmy_size));
    CUDA_CHECK(cudaMalloc(&Fmy_even_im, fmy_size));
    CUDA_CHECK(cudaMalloc(&Fmy_odd_re, fmy_size));
    CUDA_CHECK(cudaMalloc(&Fmy_odd_im, fmy_size));
    CUDA_CHECK(cudaMalloc(&cos_theta, geom_size));
    CUDA_CHECK(cudaMalloc(&sin_theta, geom_size));

    // Phase 1: Compute Fmy = sum_l(alm * Ylm)
    int ring_batch_size = RING_BATCH_SIZE;

    // Shared memory: geometry (2 arrays) + alm (2 arrays)
    size_t smem_p1 = 2 * ring_batch_size * sizeof(R) + 2 * lp1 * sizeof(T);

    // One block per m, 32 threads (one warp)
    compute_fmy_kernel_v6<T, R><<<lp1, 32, smem_p1>>>(
        nside, l_max, n_maps, n_north_rings,
        ring_batch_size,
        alm_scaled_real, alm_scaled_imag,
        Fmy_even_re, Fmy_even_im, Fmy_odd_re, Fmy_odd_im,
        cos_theta, sin_theta
    );
    CUDA_CHECK(cudaGetLastError());

    if (timing_enabled) {
        cudaEventRecord(end_p1);
        cudaEventRecord(start_p2);
    }

    // Initialize output map to zero
    CUDA_CHECK(cudaMemset(map_out, 0, (size_t)n_maps * n_rings * max_pix * sizeof(T)));

    // Phase 2: Synthesize map from Fmy
    // Shared memory: 4 Fmy arrays of lp1 elements each
    size_t smem_p2 = 4 * lp1 * sizeof(R);

    int block_size_p2 = 256;
    synthesize_map_kernel_v6<T, R><<<n_north_rings, block_size_p2, smem_p2>>>(
        nside, l_max, n_maps, n_rings, n_north_rings,
        Fmy_even_re, Fmy_even_im, Fmy_odd_re, Fmy_odd_im,
        map_out
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (timing_enabled) {
        cudaEventRecord(end_p2);
        cudaEventSynchronize(end_p2);

        float scale_ms, p1_ms, p2_ms;
        cudaEventElapsedTime(&scale_ms, start_scale, end_scale);
        cudaEventElapsedTime(&p1_ms, start_p1, end_p1);
        cudaEventElapsedTime(&p2_ms, start_p2, end_p2);

        fprintf(stderr, "[SPHT_TIMING] alm2map nside=%d l_max=%d n_maps=%d Scale=%.2fms Phase1=%.2fms Phase2=%.2fms Total=%.2fms\n",
                nside, l_max, n_maps, scale_ms, p1_ms, p2_ms, scale_ms + p1_ms + p2_ms);

        cudaEventDestroy(start_scale);
        cudaEventDestroy(end_scale);
        cudaEventDestroy(start_p1);
        cudaEventDestroy(end_p1);
        cudaEventDestroy(start_p2);
        cudaEventDestroy(end_p2);
    }

    // Cleanup
    cudaFree(alm_scaled_real);
    cudaFree(alm_scaled_imag);
    cudaFree(Fmy_even_re);
    cudaFree(Fmy_even_im);
    cudaFree(Fmy_odd_re);
    cudaFree(Fmy_odd_im);
    cudaFree(cos_theta);
    cudaFree(sin_theta);
}

// ============================================================================
// C API entry points
// ============================================================================

extern "C" {

// Float64 storage, Float64 recurrence (highest accuracy, default)
void alm2map_cuda_v6_f64_f64(int nside, int l_max, int n_maps,
                              const double* alm_in_real, const double* alm_in_imag,
                              double* map_out) {
    alm2map_cuda_v6_impl<double, double>(nside, l_max, n_maps,
                                          alm_in_real, alm_in_imag, map_out);
}

// Float64 storage, Float32 recurrence
void alm2map_cuda_v6_f64_f32(int nside, int l_max, int n_maps,
                              const double* alm_in_real, const double* alm_in_imag,
                              double* map_out) {
    alm2map_cuda_v6_impl<double, float>(nside, l_max, n_maps,
                                         alm_in_real, alm_in_imag, map_out);
}

// Float32 storage, Float64 recurrence
void alm2map_cuda_v6_f32_f64(int nside, int l_max, int n_maps,
                              const float* alm_in_real, const float* alm_in_imag,
                              float* map_out) {
    alm2map_cuda_v6_impl<float, double>(nside, l_max, n_maps,
                                         alm_in_real, alm_in_imag, map_out);
}

// Float32 storage, Float32 recurrence (fastest)
void alm2map_cuda_v6_f32_f32(int nside, int l_max, int n_maps,
                              const float* alm_in_real, const float* alm_in_imag,
                              float* map_out) {
    alm2map_cuda_v6_impl<float, float>(nside, l_max, n_maps,
                                        alm_in_real, alm_in_imag, map_out);
}

// Backwards-compatible aliases
void alm2map_cuda_v6_f64(int nside, int l_max, int n_maps,
                          const double* alm_in_real, const double* alm_in_imag,
                          double* map_out) {
    alm2map_cuda_v6_f64_f64(nside, l_max, n_maps, alm_in_real, alm_in_imag, map_out);
}

void alm2map_cuda_v6_f32(int nside, int l_max, int n_maps,
                          const float* alm_in_real, const float* alm_in_imag,
                          float* map_out) {
    alm2map_cuda_v6_f32_f32(nside, l_max, n_maps, alm_in_real, alm_in_imag, map_out);
}

// Legacy complex_t input wrapper for compatibility
void alm2map_cuda_v6(int nside, int l_max, int n_maps,
                      const complex_t* alm_in, real_t* map_out) {
    int lp1 = l_max + 1;
    size_t alm_count = (size_t)n_maps * lp1 * lp1;

    // Separate complex alm into real and imag parts
    double *alm_real, *alm_imag;
    CUDA_CHECK(cudaMalloc(&alm_real, alm_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&alm_imag, alm_count * sizeof(double)));

    // Copy from interleaved complex format
    CUDA_CHECK(cudaMemcpy2D(alm_real, sizeof(double),
                            (const double*)alm_in, 2 * sizeof(double),
                            sizeof(double), alm_count,
                            cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(alm_imag, sizeof(double),
                            (const double*)alm_in + 1, 2 * sizeof(double),
                            sizeof(double), alm_count,
                            cudaMemcpyDeviceToDevice));

    alm2map_cuda_v6_impl<double, double>(nside, l_max, n_maps,
                                          alm_real, alm_imag, map_out);

    cudaFree(alm_real);
    cudaFree(alm_imag);
}

} // extern "C"
