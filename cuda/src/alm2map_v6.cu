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
#include "../include/bluestein_fft.h"
#include <stdio.h>
#include <type_traits>

// ============================================================================
// Runtime Phase 1 method selection (defined in map2alm_v6.cu)
// ============================================================================

enum class Phase1Method {
    DFT = 0,
    FFT_EQUATORIAL = 1,
    BLUESTEIN = 2
};

// External reference to global flag from map2alm_v6.cu
extern Phase1Method g_phase1_method;

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

    // Precompute m-dependent values (used throughout kernel)
    C m2_precomp = C(m * m);
    C recur_c_m1 = Traits::sqrt_d(C(2*m + 3));  // For l = m+1 recurrence

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
                        Ylm = cos_th * recur_c_m1 * Ylm_p1;
                        Ylm_p2 = Ylm_p1;
                        Ylm_p1 = Ylm;
                    } else {
                        C l2 = C(l * l);
                        C lm1_2 = C((l-1) * (l-1));

                        C A = Traits::sqrt_d((C(4)*l2 - C(1)) / (l2 - m2_precomp));
                        C B = Traits::sqrt_d((C(2*l + 1)) / (C(2*l - 3)) * (lm1_2 - m2_precomp) / (l2 - m2_precomp));

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

// Spin-2 scaling kernel: applies both m>0 factor (×2) AND norm(l) pre-scaling
// norm(l) = 1/sqrt((l-1)*l*(l+1)*(l+2)) for l >= 2, 0 otherwise
template<typename T>
__global__ void scale_alm_spin2_for_synth_kernel(
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

    // Compute combined scale: (m>0 ? 2 : 1) × norm(l)
    T scale;
    if (l < 2) {
        scale = T(0);  // No spin-2 contribution for l < 2
    } else {
        T m_factor = (m > 0) ? T(2.0) : T(1.0);
        // norm = 1/sqrt((l-1)*l*(l+1)*(l+2))
        T prod = T((l-1) * l) * T((l+1) * (l+2));
        T norm = T(1.0) / sqrt(prod);
        scale = m_factor * norm;
    }
    alm_out_real[idx] = alm_in_real[idx] * scale;
    alm_out_imag[idx] = alm_in_imag[idx] * scale;
}

// ============================================================================
// Kernel to convert Fmy from even/odd to north/south form
// Fmy_north = (Fmy_even + Fmy_odd) / 2
// Fmy_south = (Fmy_even - Fmy_odd) / 2
// ============================================================================

template<typename R>
__global__ void convert_fmy_even_odd_to_north_south_kernel(
    int n_maps, int lp1, int n_north_rings,
    const R* __restrict__ Fmy_even_re,
    const R* __restrict__ Fmy_even_im,
    const R* __restrict__ Fmy_odd_re,
    const R* __restrict__ Fmy_odd_im,
    R* __restrict__ Fmy_north_re,
    R* __restrict__ Fmy_north_im,
    R* __restrict__ Fmy_south_re,
    R* __restrict__ Fmy_south_im
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n_maps * lp1 * n_north_rings;

    if (idx >= total) return;

    R even_re = Fmy_even_re[idx];
    R even_im = Fmy_even_im[idx];
    R odd_re = Fmy_odd_re[idx];
    R odd_im = Fmy_odd_im[idx];

    Fmy_north_re[idx] = R(0.5) * (even_re + odd_re);
    Fmy_north_im[idx] = R(0.5) * (even_im + odd_im);
    Fmy_south_re[idx] = R(0.5) * (even_re - odd_re);
    Fmy_south_im[idx] = R(0.5) * (even_im - odd_im);
}

// ============================================================================
// Bluestein inverse pre-chirp kernel for alm2map
// Handles both north and south hemispheres using separate Fmy arrays
// ============================================================================

template<typename R>
__global__ void bluestein_alm2map_pre_chirp_kernel(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const R* __restrict__ Fmy_north_re,  // [n_maps, lp1, n_north_rings]
    const R* __restrict__ Fmy_north_im,
    const R* __restrict__ Fmy_south_re,
    const R* __restrict__ Fmy_south_im,
    cufftDoubleComplex* __restrict__ chirped_out,  // [n_maps, n_rings, M]
    int* __restrict__ ring_sizes_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;
    int max_pix = 4 * nside;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    // Determine if north or south and get corresponding north_ring index
    bool is_south = (ring_idx >= n_north_rings);
    int north_ring = is_south ? (n_rings - 1 - ring_idx) : ring_idx;

    // Compute ring geometry
    double cos_th, sin_th, phi0;
    int N;
    {
        int ring_i = ring_idx + 1;
        if (ring_i < nside) {
            double i2_3n2 = double(ring_i * ring_i) / double(3.0 * nside * nside);
            cos_th = 1.0 - i2_3n2;
            sin_th = sqrt(1.0 - cos_th * cos_th);
            phi0 = M_PI / (2.0 * ring_i) * 0.5;
            N = 4 * ring_i;
        } else if (ring_i > 3 * nside) {
            int mirror_i = 4 * nside - ring_i;
            double i2_3n2 = double(mirror_i * mirror_i) / double(3.0 * nside * nside);
            cos_th = -(1.0 - i2_3n2);
            sin_th = sqrt(1.0 - cos_th * cos_th);
            phi0 = M_PI / (2.0 * mirror_i) * 0.5;
            N = 4 * mirror_i;
        } else {
            cos_th = 4.0 / 3.0 - 2.0 * ring_i / (3.0 * nside);
            sin_th = sqrt(1.0 - cos_th * cos_th);
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0 = M_PI / (2.0 * nside) * (1.0 - s / 2.0);
            N = 4 * nside;
        }
    }

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    // Select appropriate Fmy array
    const R* fmy_re = is_south ? Fmy_south_re : Fmy_north_re;
    const R* fmy_im = is_south ? Fmy_south_im : Fmy_north_im;

    cufftDoubleComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;
    double pi_over_N = M_PI / double(N);

    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        cufftDoubleComplex val;
        if (m <= l_max) {
            size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            double fmy_r = double(fmy_re[idx]);
            double fmy_i = double(fmy_im[idx]);

            // Phase correction: multiply by exp(+i * m * phi0)
            double phase_angle = double(m) * phi0;
            double phase_c, phase_s;
            sincos(phase_angle, &phase_s, &phase_c);
            double fmy_r_corr = fmy_r * phase_c - fmy_i * phase_s;
            double fmy_i_corr = fmy_r * phase_s + fmy_i * phase_c;

            // Pre-chirp for IDFT: multiply by exp(+πi*m²/N)
            double chirp_angle = pi_over_N * double(m) * double(m);
            double chirp_c, chirp_s;
            sincos(chirp_angle, &chirp_s, &chirp_c);
            val.x = fmy_r_corr * chirp_c - fmy_i_corr * chirp_s;
            val.y = fmy_r_corr * chirp_s + fmy_i_corr * chirp_c;
        } else {
            val.x = 0.0;
            val.y = 0.0;
        }
        chirped[m] = val;
    }
}

// Float32 version
template<typename R>
__global__ void bluestein_alm2map_pre_chirp_kernel_f32(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const R* __restrict__ Fmy_north_re,
    const R* __restrict__ Fmy_north_im,
    const R* __restrict__ Fmy_south_re,
    const R* __restrict__ Fmy_south_im,
    cufftComplex* __restrict__ chirped_out,
    int* __restrict__ ring_sizes_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    bool is_south = (ring_idx >= n_north_rings);
    int north_ring = is_south ? (n_rings - 1 - ring_idx) : ring_idx;

    float cos_th, sin_th, phi0;
    int N;
    {
        int ring_i = ring_idx + 1;
        if (ring_i < nside) {
            float i2_3n2 = float(ring_i * ring_i) / float(3.0f * nside * nside);
            cos_th = 1.0f - i2_3n2;
            sin_th = sqrtf(1.0f - cos_th * cos_th);
            phi0 = float(M_PI) / float(2.0f * ring_i) * 0.5f;
            N = 4 * ring_i;
        } else if (ring_i > 3 * nside) {
            int mirror_i = 4 * nside - ring_i;
            float i2_3n2 = float(mirror_i * mirror_i) / float(3.0f * nside * nside);
            cos_th = -(1.0f - i2_3n2);
            sin_th = sqrtf(1.0f - cos_th * cos_th);
            phi0 = float(M_PI) / float(2.0f * mirror_i) * 0.5f;
            N = 4 * mirror_i;
        } else {
            cos_th = 4.0f / 3.0f - 2.0f * ring_i / (3.0f * nside);
            sin_th = sqrtf(1.0f - cos_th * cos_th);
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0 = float(M_PI) / float(2.0f * nside) * (1.0f - s / 2.0f);
            N = 4 * nside;
        }
    }

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    const R* fmy_re = is_south ? Fmy_south_re : Fmy_north_re;
    const R* fmy_im = is_south ? Fmy_south_im : Fmy_north_im;

    cufftComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;
    float pi_over_N = float(M_PI) / float(N);

    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        cufftComplex val;
        if (m <= l_max) {
            size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            float fmy_r = float(fmy_re[idx]);
            float fmy_i = float(fmy_im[idx]);

            float phase_angle = float(m) * phi0;
            float phase_c, phase_s;
            sincosf(phase_angle, &phase_s, &phase_c);
            float fmy_r_corr = fmy_r * phase_c - fmy_i * phase_s;
            float fmy_i_corr = fmy_r * phase_s + fmy_i * phase_c;

            float chirp_angle = pi_over_N * float(m) * float(m);
            float chirp_c, chirp_s;
            sincosf(chirp_angle, &chirp_s, &chirp_c);
            val.x = fmy_r_corr * chirp_c - fmy_i_corr * chirp_s;
            val.y = fmy_r_corr * chirp_s + fmy_i_corr * chirp_c;
        } else {
            val.x = 0.0f;
            val.y = 0.0f;
        }
        chirped[m] = val;
    }
}

// Extract map from Bluestein IFFT result
template<typename T>
__global__ void bluestein_alm2map_extract_kernel(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,
    T* __restrict__ map_out
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
__global__ void bluestein_alm2map_extract_kernel_f32(
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
// Inverse chirp kernels for IDFT (different sign from forward DFT)
// For IDFT, the convolution chirp is h[j] = exp(-πi*j²/N) (negative sign)
// ============================================================================

__global__ void bluestein_compute_inv_chirp_kernel(
    int nside, int l_max, int M,
    cufftDoubleComplex* __restrict__ inv_chirp_fft  // [nside, M]
) {
    int size_idx = blockIdx.x;  // 0 = size 4, 1 = size 8, ..., nside-1 = size 4*nside
    int N = 4 * (size_idx + 1);

    if (N > 4 * nside) return;

    cufftDoubleComplex* chirp = inv_chirp_fft + size_idx * M;
    double pi_over_N = M_PI / double(N);

    // For Bluestein with input length K = l_max + 1 and output length N,
    // we need chirp values at indices 0..K-1 and M-K+1..M-1 (for negative wrap)
    // Use K = max(N, l_max + 1) to cover all needed indices
    int K = (l_max + 1 > N) ? (l_max + 1) : N;

    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        cufftDoubleComplex val;
        int j_eff;

        if (j < K) {
            j_eff = j;
        } else if (j >= M - K + 1) {
            j_eff = M - j;  // Wrap-around for negative indices (gives negative j_eff)
        } else {
            val.x = 0.0;
            val.y = 0.0;
            chirp[j] = val;
            continue;
        }

        // NEGATIVE sign for IDFT: exp(-πi*j_eff²/N)
        // Note: j_eff can be negative for wrap-around indices
        double angle = pi_over_N * double(j_eff) * double(j_eff);
        double c, s;
        sincos(angle, &s, &c);
        val.x = c;
        val.y = -s;  // Note: negative sin for exp(-i*angle)
        chirp[j] = val;
    }
}

__global__ void bluestein_compute_inv_chirp_kernel_f32(
    int nside, int l_max, int M,
    cufftComplex* __restrict__ inv_chirp_fft
) {
    int size_idx = blockIdx.x;
    int N = 4 * (size_idx + 1);

    if (N > 4 * nside) return;

    cufftComplex* chirp = inv_chirp_fft + size_idx * M;
    float pi_over_N = float(M_PI) / float(N);

    // For Bluestein with input length K = l_max + 1 and output length N,
    // we need chirp values at indices 0..K-1 and M-K+1..M-1
    int K = (l_max + 1 > N) ? (l_max + 1) : N;

    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        cufftComplex val;
        int j_eff;

        if (j < K) {
            j_eff = j;
        } else if (j >= M - K + 1) {
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
        val.y = -s;  // Negative sin for IDFT
        chirp[j] = val;
    }
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
    // Method selected by g_phase1_method (DFT or Bluestein)

    if (g_phase1_method == Phase1Method::BLUESTEIN) {
        // ============================================================
        // BLUESTEIN INVERSE FFT: Fmy -> map via chirp-z transform
        // ============================================================

        // First convert Fmy from even/odd to north/south form
        R *Fmy_north_re, *Fmy_north_im, *Fmy_south_re, *Fmy_south_im;
        CUDA_CHECK(cudaMalloc(&Fmy_north_re, fmy_size));
        CUDA_CHECK(cudaMalloc(&Fmy_north_im, fmy_size));
        CUDA_CHECK(cudaMalloc(&Fmy_south_re, fmy_size));
        CUDA_CHECK(cudaMalloc(&Fmy_south_im, fmy_size));

        size_t total_elements = (size_t)n_maps * lp1 * n_north_rings;
        int block_conv = 256;
        int grid_conv = (total_elements + block_conv - 1) / block_conv;
        convert_fmy_even_odd_to_north_south_kernel<R><<<grid_conv, block_conv>>>(
            n_maps, lp1, n_north_rings,
            Fmy_even_re, Fmy_even_im, Fmy_odd_re, Fmy_odd_im,
            Fmy_north_re, Fmy_north_im, Fmy_south_re, Fmy_south_im
        );
        CUDA_CHECK(cudaGetLastError());

        // Compute M = next power of 2 >= l_max + 1 + max_ring_size - 1
        // This is required because Bluestein convolution needs M >= input_length + output_length - 1
        // where input_length = l_max + 1 (number of m values) and output_length = ring_size
        int max_ring_size = 4 * nside;
        int M = next_power_of_2(lp1 + max_ring_size - 1);

        // Allocate Bluestein buffers
        int* ring_sizes;
        CUDA_CHECK(cudaMalloc(&ring_sizes, n_rings * sizeof(int)));

        bool use_double = std::is_same<R, double>::value;

        if (use_double) {
            // Double precision Bluestein inverse
            cufftDoubleComplex* chirped_data;
            cufftDoubleComplex* conj_chirp_fft;

            size_t chirp_data_size = (size_t)n_maps * n_rings * M * sizeof(cufftDoubleComplex);
            size_t conj_chirp_size = (size_t)nside * M * sizeof(cufftDoubleComplex);

            CUDA_CHECK(cudaMalloc(&chirped_data, chirp_data_size));
            CUDA_CHECK(cudaMalloc(&conj_chirp_fft, conj_chirp_size));

            // Compute conjugate chirp for all unique ring sizes
            bluestein_compute_inv_chirp_kernel<<<nside, 256>>>(nside, l_max, M, conj_chirp_fft);
            CUDA_CHECK(cudaGetLastError());

            // FFT the conjugate chirps
            cufftHandle chirp_fft_plan = get_cached_fft_plan(M, nside, CUFFT_Z2Z);
            cufftExecZ2Z(chirp_fft_plan, conj_chirp_fft, conj_chirp_fft, CUFFT_FORWARD);

            // Pre-chirp for all rings (north and south)
            dim3 grid_all(n_rings, n_maps);
            bluestein_alm2map_pre_chirp_kernel<R><<<grid_all, 256>>>(
                nside, n_maps, n_rings, n_north_rings, l_max, M,
                Fmy_north_re, Fmy_north_im, Fmy_south_re, Fmy_south_im,
                chirped_data, ring_sizes
            );
            CUDA_CHECK(cudaGetLastError());

            // FFT all chirped data
            cufftHandle data_fft_plan = get_cached_fft_plan(M, n_maps * n_rings, CUFFT_Z2Z);
            cufftExecZ2Z(data_fft_plan, chirped_data, chirped_data, CUFFT_FORWARD);

            // Pointwise multiply with conjugate chirp
            bluestein_pointwise_mult_kernel_v2<<<grid_all, 256>>>(
                n_maps, n_rings, M, ring_sizes, chirped_data, conj_chirp_fft
            );
            CUDA_CHECK(cudaGetLastError());

            // IFFT
            cufftExecZ2Z(data_fft_plan, chirped_data, chirped_data, CUFFT_INVERSE);

            // Extract map pixels with post-chirp
            bluestein_alm2map_extract_kernel<T><<<grid_all, 256>>>(
                nside, n_maps, n_rings, M,
                ring_sizes, chirped_data, map_out
            );
            CUDA_CHECK(cudaGetLastError());

            cudaFree(chirped_data);
            cudaFree(conj_chirp_fft);
        } else {
            // Float32 precision Bluestein inverse
            cufftComplex* chirped_data;
            cufftComplex* conj_chirp_fft;

            size_t chirp_data_size = (size_t)n_maps * n_rings * M * sizeof(cufftComplex);
            size_t conj_chirp_size = (size_t)nside * M * sizeof(cufftComplex);

            CUDA_CHECK(cudaMalloc(&chirped_data, chirp_data_size));
            CUDA_CHECK(cudaMalloc(&conj_chirp_fft, conj_chirp_size));

            bluestein_compute_inv_chirp_kernel_f32<<<nside, 256>>>(nside, l_max, M, conj_chirp_fft);
            CUDA_CHECK(cudaGetLastError());

            cufftHandle chirp_fft_plan = get_cached_fft_plan(M, nside, CUFFT_C2C);
            cufftExecC2C(chirp_fft_plan, conj_chirp_fft, conj_chirp_fft, CUFFT_FORWARD);

            dim3 grid_all(n_rings, n_maps);
            bluestein_alm2map_pre_chirp_kernel_f32<R><<<grid_all, 256>>>(
                nside, n_maps, n_rings, n_north_rings, l_max, M,
                Fmy_north_re, Fmy_north_im, Fmy_south_re, Fmy_south_im,
                chirped_data, ring_sizes
            );
            CUDA_CHECK(cudaGetLastError());

            cufftHandle data_fft_plan = get_cached_fft_plan(M, n_maps * n_rings, CUFFT_C2C);
            cufftExecC2C(data_fft_plan, chirped_data, chirped_data, CUFFT_FORWARD);

            bluestein_pointwise_mult_kernel_f32_v2<<<grid_all, 256>>>(
                n_maps, n_rings, M, ring_sizes, chirped_data, conj_chirp_fft
            );
            CUDA_CHECK(cudaGetLastError());

            cufftExecC2C(data_fft_plan, chirped_data, chirped_data, CUFFT_INVERSE);

            bluestein_alm2map_extract_kernel_f32<T><<<grid_all, 256>>>(
                nside, n_maps, n_rings, M,
                ring_sizes, chirped_data, map_out
            );
            CUDA_CHECK(cudaGetLastError());

            cudaFree(chirped_data);
            cudaFree(conj_chirp_fft);
        }

        cudaFree(ring_sizes);
        cudaFree(Fmy_north_re);
        cudaFree(Fmy_north_im);
        cudaFree(Fmy_south_re);
        cudaFree(Fmy_south_im);

    } else {
        // ============================================================
        // DIRECT DFT: Default method
        // ============================================================
        size_t smem_p2 = 4 * lp1 * sizeof(R);
        int block_size_p2 = 256;

        // Request extended shared memory if needed
        const size_t MAX_SMEM = 48 * 1024;
        if (smem_p2 > MAX_SMEM) {
            cudaError_t attr_err = cudaFuncSetAttribute(
                synthesize_map_kernel_v6<T, R>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_p2);
            if (attr_err != cudaSuccess) {
                fprintf(stderr, "Error: alm2map Phase 2 requires %zu bytes shared memory "
                        "(l_max=%d), but GPU limit exceeded.\n"
                        "Try using float32 storage precision for large nside.\n",
                        smem_p2, l_max);
                // Clean up
                cudaFree(alm_scaled_real); cudaFree(alm_scaled_imag);
                cudaFree(Fmy_even_re); cudaFree(Fmy_even_im);
                cudaFree(Fmy_odd_re); cudaFree(Fmy_odd_im);
                cudaFree(cos_theta); cudaFree(sin_theta);
                return;
            }
        }

        synthesize_map_kernel_v6<T, R><<<n_north_rings, block_size_p2, smem_p2>>>(
            nside, l_max, n_maps, n_rings, n_north_rings,
            Fmy_even_re, Fmy_even_im, Fmy_odd_re, Fmy_odd_im,
            map_out
        );
        CUDA_CHECK(cudaGetLastError());
    }

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

// ============================================================================
// SPIN-2 IMPLEMENTATION
// ============================================================================
// Spin-2 transforms for polarization (E,B) alm -> (Q,U) maps
// Reference: jax_healpix/SPHT_jax.py alm2map() and alm2ring_ns()
//
// alm2map spin-2 (from SPHT_jax.py lines 323-342):
//   Q = ₂Y×E + i×₋₂Y×B
//   U = ₋₂Y×E + i×₂Y×B
//   Post-processing: U *= i, Q *= -1
//
// South ring symmetry:
//   ₂Y(-θ) = (-1)^(l+m) × ₂Y(θ)
//   ₋₂Y(-θ) = (-1)^(l+m+1) × ₋₂Y(θ)  [extra sign flip!]
// ============================================================================

// Compute spin-2 normalization factor: sqrt((l-2)!/(l+2)!)
template<typename C>
__device__ __forceinline__ C compute_spin2_norm_synth(int l) {
    if (l < 2) return C(0);
    C prod = C((l-1) * l) * C((l+1) * (l+2));
    return C(1.0) / sqrt(prod);
}

// Compute alpha_{l,m} = sqrt((2l+1)(l²-m²)/(2l-1))
template<typename C>
__device__ __forceinline__ C compute_alpha_lm_synth(int l, int m) {
    if (l <= 1) return C(0);
    C l2 = C(l * l);
    C m2 = C(m * m);
    return sqrt(C(2*l + 1) * (l2 - m2) / C(2*l - 1));
}

// ============================================================================
// Spin-2 Phase 1: Compute Fmy for Q and U maps from E,B alm
// Fmy_Q = ₂Y×E + i×₋₂Y×B  → Fmy_Q_re = ₂Y×E, Fmy_Q_im = ₋₂Y×B
// Fmy_U = ₋₂Y×E + i×₂Y×B  → Fmy_U_re = ₋₂Y×E, Fmy_U_im = ₂Y×B
// ============================================================================

template<typename T, typename R>
__global__ void compute_fmy_spin2_kernel_v6(
    int nside, int l_max, int n_maps, int n_north_rings,
    int ring_batch_size,
    const T* __restrict__ alm_E_re,  // [n_maps, lp1, lp1]
    const T* __restrict__ alm_E_im,
    const T* __restrict__ alm_B_re,
    const T* __restrict__ alm_B_im,
    R* __restrict__ Fmy_Q_north_re,  // [n_maps, lp1, n_north_rings]
    R* __restrict__ Fmy_Q_north_im,
    R* __restrict__ Fmy_Q_south_re,
    R* __restrict__ Fmy_Q_south_im,
    R* __restrict__ Fmy_U_north_re,
    R* __restrict__ Fmy_U_north_im,
    R* __restrict__ Fmy_U_south_re,
    R* __restrict__ Fmy_U_south_im,
    R* __restrict__ cos_theta_out,
    R* __restrict__ sin_theta_out
) {
    using Traits = V6TraitsSynth<R>;
    using C = typename Traits::compute_t;

    int m = blockIdx.x;
    int lane = threadIdx.x;
    int lp1 = l_max + 1;

    if (m > l_max || lane >= 32) return;

    // Precompute m-dependent values (used throughout kernel)
    C m2_precomp = C(m * m);
    C two_m_precomp = C(2 * m);
    C recur_c_m1 = Traits::sqrt_d(C(2*m + 3));  // For l = m+1 recurrence

    extern __shared__ char smem[];
    R* sh_cos_th = (R*)smem;
    R* sh_sin_th = sh_cos_th + ring_batch_size;
    T* sh_alm_E_re = (T*)(sh_sin_th + ring_batch_size);
    T* sh_alm_E_im = sh_alm_E_re + lp1;
    T* sh_alm_B_re = sh_alm_E_im + lp1;
    T* sh_alm_B_im = sh_alm_B_re + lp1;

    C Ylm_prev1[MAX_RINGS_PER_LANE];
    C Ylm_prev2[MAX_RINGS_PER_LANE];

    int n_my_rings_total = (n_north_rings + 31 - lane) / 32;

    for (int batch_start = 0; batch_start < n_north_rings; batch_start += ring_batch_size) {
        int batch_end = min(batch_start + ring_batch_size, n_north_rings);
        int batch_size = batch_end - batch_start;

        // Load geometry
        if (m == 0) {
            for (int r = lane; r < batch_size; r += 32) {
                int global_r = batch_start + r;
                T cos_th, sin_th, phi0;
                int npix;
                compute_ring_geom_synth_v6<T>(global_r, nside, &cos_th, &sin_th, &phi0, &npix);
                sh_cos_th[r] = R(cos_th);
                sh_sin_th[r] = R(sin_th);
                if (lane == 0 || r == lane) {
                    cos_theta_out[global_r] = R(cos_th);
                    sin_theta_out[global_r] = R(sin_th);
                }
            }
        } else {
            for (int r = lane; r < batch_size; r += 32) {
                int global_r = batch_start + r;
                T cos_th, sin_th, phi0;
                int npix;
                compute_ring_geom_synth_v6<T>(global_r, nside, &cos_th, &sin_th, &phi0, &npix);
                sh_cos_th[r] = R(cos_th);
                sh_sin_th[r] = R(sin_th);
            }
        }

        int k_start = (batch_start > lane) ? (batch_start - lane + 31) / 32 : 0;
        int k_end = (batch_end > lane) ? (batch_end - 1 - lane) / 32 + 1 : 0;
        k_end = min(k_end, n_my_rings_total);

        // Initialize Y[m,m]
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

        for (int t = 0; t < n_maps; t++) {
            const T* alm_E_re_t = alm_E_re + (size_t)t * lp1 * lp1;
            const T* alm_E_im_t = alm_E_im + (size_t)t * lp1 * lp1;
            const T* alm_B_re_t = alm_B_re + (size_t)t * lp1 * lp1;
            const T* alm_B_im_t = alm_B_im + (size_t)t * lp1 * lp1;

            // Load alm values (scaled by 2 for m > 0)
            for (int l = m + lane; l <= l_max; l += 32) {
                sh_alm_E_re[l] = alm_E_re_t[l * lp1 + m];
                sh_alm_E_im[l] = alm_E_im_t[l * lp1 + m];
                sh_alm_B_re[l] = alm_B_re_t[l * lp1 + m];
                sh_alm_B_im[l] = alm_B_im_t[l * lp1 + m];
            }
            __syncwarp();

            for (int k = k_start; k < k_end; k++) {
                int global_r = lane + 32 * k;
                int local_r = global_r - batch_start;
                C cos_th = C(sh_cos_th[local_r]);
                C sin_th = C(sh_sin_th[local_r]);
                C sin_th_sq = sin_th * sin_th;
                C inv_sin_sq = (sin_th_sq > C(1e-20)) ? C(1.0) / sin_th_sq : C(0);

                // Accumulators for Fmy components
                C fmy_Q_n_re = C(0), fmy_Q_n_im = C(0);
                C fmy_Q_s_re = C(0), fmy_Q_s_im = C(0);
                C fmy_U_n_re = C(0), fmy_U_n_im = C(0);
                C fmy_U_s_re = C(0), fmy_U_s_im = C(0);

                // Recompute Ylm for this ring
                C Ymm = C(1.0) / Traits::sqrt_d(C(4.0 * Traits::PI_VAL));
                for (int j = 1; j <= m; j++) {
                    Ymm *= -sin_th * Traits::sqrt_d(C(2*j + 1) / C(2*j));
                }

                // Advance recurrence from l=m to l=l_start-1 for spin-2
                int l_start = max(2, m);
                C Ylm_p1 = Ymm;
                C Ylm_p2 = C(0);

                for (int l = m + 1; l < l_start; l++) {
                    if (l == m + 1) {
                        C Yl_new = cos_th * recur_c_m1 * Ylm_p1;
                        Ylm_p2 = Ylm_p1;
                        Ylm_p1 = Yl_new;
                    } else {
                        C l2 = C(l * l);
                        C lm1_2 = C((l-1) * (l-1));
                        C A = Traits::sqrt_d((C(4)*l2 - C(1)) / (l2 - m2_precomp));
                        C B = Traits::sqrt_d((C(2*l + 1)) / (C(2*l - 3)) * (lm1_2 - m2_precomp) / (l2 - m2_precomp));
                        C Yl_new = A * cos_th * Ylm_p1 - B * Ylm_p2;
                        Ylm_p2 = Ylm_p1;
                        Ylm_p1 = Yl_new;
                    }
                }

                // Process l from l_start to l_max
                for (int l = l_start; l <= l_max; l++) {
                    C Ylm, Ylm_prev;
                    if (l == m) {
                        Ylm = Ymm;
                        Ylm_prev = C(0);
                    } else if (l == m + 1) {
                        Ylm = cos_th * recur_c_m1 * Ylm_p1;
                        Ylm_prev = Ylm_p1;
                        Ylm_p2 = Ylm_p1;
                        Ylm_p1 = Ylm;
                    } else {
                        C l2 = C(l * l);
                        C lm1_2 = C((l-1) * (l-1));
                        C A = Traits::sqrt_d((C(4)*l2 - C(1)) / (l2 - m2_precomp));
                        C B = Traits::sqrt_d((C(2*l + 1)) / (C(2*l - 3)) * (lm1_2 - m2_precomp) / (l2 - m2_precomp));
                        Ylm = A * cos_th * Ylm_p1 - B * Ylm_p2;
                        Ylm_prev = Ylm_p1;
                        Ylm_p2 = Ylm_p1;
                        Ylm_p1 = Ylm;
                    }

                    // Compute spin-2 harmonics (norm already pre-scaled into alm)
                    C alpha = compute_alpha_lm_synth<C>(l, m);
                    C ll1 = C(l * (l - 1));

                    C coeff1 = (C(2) * (m2_precomp - C(l)) * inv_sin_sq - ll1);
                    C coeff2 = C(2) * alpha * cos_th * inv_sin_sq;
                    C Y2 = coeff1 * Ylm + coeff2 * Ylm_prev;

                    C inner = alpha * Ylm_prev - C(l - 1) * cos_th * Ylm;
                    C Ym2 = two_m_precomp * inv_sin_sq * inner;

                    // Get alm values
                    C alm_E_r = C(sh_alm_E_re[l]);
                    C alm_E_i = C(sh_alm_E_im[l]);
                    C alm_B_r = C(sh_alm_B_re[l]);
                    C alm_B_i = C(sh_alm_B_im[l]);

                    // Fmy_Q = ₂Y×E + i×₋₂Y×B
                    // Fmy_Q_re = ₂Y×E_re - ₋₂Y×B_im, Fmy_Q_im = ₂Y×E_im + ₋₂Y×B_re
                    C fQ_re = Y2 * alm_E_r - Ym2 * alm_B_i;
                    C fQ_im = Y2 * alm_E_i + Ym2 * alm_B_r;

                    // Fmy_U = ₋₂Y×E + i×₂Y×B
                    C fU_re = Ym2 * alm_E_r - Y2 * alm_B_i;
                    C fU_im = Ym2 * alm_E_i + Y2 * alm_B_r;

                    // North ring
                    fmy_Q_n_re += fQ_re;
                    fmy_Q_n_im += fQ_im;
                    fmy_U_n_re += fU_re;
                    fmy_U_n_im += fU_im;

                    // South ring: different parity for +2 and -2 spins
                    int parity_p2 = (l + m) & 1;
                    int parity_m2 = (l + m + 1) & 1;
                    C sign_Y2 = parity_p2 ? C(-1) : C(1);
                    C sign_Ym2 = parity_m2 ? C(-1) : C(1);

                    C Y2_s = sign_Y2 * Y2;
                    C Ym2_s = sign_Ym2 * Ym2;

                    C fQ_s_re = Y2_s * alm_E_r - Ym2_s * alm_B_i;
                    C fQ_s_im = Y2_s * alm_E_i + Ym2_s * alm_B_r;
                    C fU_s_re = Ym2_s * alm_E_r - Y2_s * alm_B_i;
                    C fU_s_im = Ym2_s * alm_E_i + Y2_s * alm_B_r;

                    fmy_Q_s_re += fQ_s_re;
                    fmy_Q_s_im += fQ_s_im;
                    fmy_U_s_re += fU_s_re;
                    fmy_U_s_im += fU_s_im;
                }

                // Store north/south Fmy directly (no even/odd combination)
                size_t idx = (size_t)t * lp1 * n_north_rings + (size_t)m * n_north_rings + global_r;
                Fmy_Q_north_re[idx] = R(fmy_Q_n_re);
                Fmy_Q_north_im[idx] = R(fmy_Q_n_im);
                Fmy_Q_south_re[idx] = R(fmy_Q_s_re);
                Fmy_Q_south_im[idx] = R(fmy_Q_s_im);
                Fmy_U_north_re[idx] = R(fmy_U_n_re);
                Fmy_U_north_im[idx] = R(fmy_U_n_im);
                Fmy_U_south_re[idx] = R(fmy_U_s_re);
                Fmy_U_south_im[idx] = R(fmy_U_s_im);
            }

            __syncwarp();
        }
    }
}

// ============================================================================
// Spin-2 Phase 2: Synthesize Q,U maps from Fmy via inverse DFT
// Note: For spin-2, the output is complex-valued before post-processing
// ============================================================================

template<typename T, typename R>
__global__ void synthesize_map_spin2_kernel_v6(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings,
    const R* __restrict__ Fmy_Q_north_re,
    const R* __restrict__ Fmy_Q_north_im,
    const R* __restrict__ Fmy_Q_south_re,
    const R* __restrict__ Fmy_Q_south_im,
    const R* __restrict__ Fmy_U_north_re,
    const R* __restrict__ Fmy_U_north_im,
    const R* __restrict__ Fmy_U_south_re,
    const R* __restrict__ Fmy_U_south_im,
    T* __restrict__ map_Q_out,    // Real output (after post-processing)
    T* __restrict__ map_U_out
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

    // Shared memory for Fmy values - only 4 arrays at a time (Q/U × re/im)
    // Process north and south separately to halve shared memory requirement
    extern __shared__ char shared_mem[];
    R* sh_fmy_Q_re = (R*)shared_mem;
    R* sh_fmy_Q_im = sh_fmy_Q_re + lp1;
    R* sh_fmy_U_re = sh_fmy_Q_im + lp1;
    R* sh_fmy_U_im = sh_fmy_U_re + lp1;

    for (int t = 0; t < n_maps; t++) {
        T* map_Q_t = map_Q_out + (size_t)t * n_rings * max_pix;
        T* map_U_t = map_U_out + (size_t)t * n_rings * max_pix;

        // === North ring pass ===
        // Load Fmy_north directly (no arithmetic needed)
        for (int m = tid; m <= l_max; m += block_size) {
            size_t idx = (size_t)t * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            sh_fmy_Q_re[m] = Fmy_Q_north_re[idx];
            sh_fmy_Q_im[m] = Fmy_Q_north_im[idx];
            sh_fmy_U_re[m] = Fmy_U_north_re[idx];
            sh_fmy_U_im[m] = Fmy_U_north_im[idx];
        }
        __syncthreads();

        // Synthesize north ring pixels
        for (int j = tid; j < n_pix_n; j += block_size) {
            C sum_Q_re = C(0), sum_Q_im = C(0);
            C sum_U_re = C(0), sum_U_im = C(0);

            C phi_j = C(phi0_n) + C(j) * C(2.0 * Traits::PI_VAL) / C(n_pix_n);

            for (int m = 0; m <= l_max; m++) {
                C fmy_Q_re = C(sh_fmy_Q_re[m]);
                C fmy_Q_im = C(sh_fmy_Q_im[m]);
                C fmy_U_re = C(sh_fmy_U_re[m]);
                C fmy_U_im = C(sh_fmy_U_im[m]);

                C angle = C(m) * phi_j;
                C cos_ang, sin_ang;
                Traits::sincos_d(angle, &sin_ang, &cos_ang);

                sum_Q_re += fmy_Q_re * cos_ang - fmy_Q_im * sin_ang;
                sum_Q_im += fmy_Q_re * sin_ang + fmy_Q_im * cos_ang;
                sum_U_re += fmy_U_re * cos_ang - fmy_U_im * sin_ang;
                sum_U_im += fmy_U_re * sin_ang + fmy_U_im * cos_ang;
            }

            map_Q_t[north_ring * max_pix + j] = T(sum_Q_re);
            map_U_t[north_ring * max_pix + j] = T(sum_U_re);
        }
        __syncthreads();

        // === South ring pass ===
        if (!is_equator) {
            // Load Fmy_south directly (no arithmetic needed)
            for (int m = tid; m <= l_max; m += block_size) {
                size_t idx = (size_t)t * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
                sh_fmy_Q_re[m] = Fmy_Q_south_re[idx];
                sh_fmy_Q_im[m] = Fmy_Q_south_im[idx];
                sh_fmy_U_re[m] = Fmy_U_south_re[idx];
                sh_fmy_U_im[m] = Fmy_U_south_im[idx];
            }
            __syncthreads();

            // Synthesize south ring pixels
            for (int j = tid; j < n_pix_s; j += block_size) {
                C sum_Q_re = C(0), sum_Q_im = C(0);
                C sum_U_re = C(0), sum_U_im = C(0);

                C phi_j = C(phi0_s) + C(j) * C(2.0 * Traits::PI_VAL) / C(n_pix_s);

                for (int m = 0; m <= l_max; m++) {
                    C fmy_Q_re = C(sh_fmy_Q_re[m]);
                    C fmy_Q_im = C(sh_fmy_Q_im[m]);
                    C fmy_U_re = C(sh_fmy_U_re[m]);
                    C fmy_U_im = C(sh_fmy_U_im[m]);

                    C angle = C(m) * phi_j;
                    C cos_ang, sin_ang;
                    Traits::sincos_d(angle, &sin_ang, &cos_ang);

                    sum_Q_re += fmy_Q_re * cos_ang - fmy_Q_im * sin_ang;
                    sum_Q_im += fmy_Q_re * sin_ang + fmy_Q_im * cos_ang;
                    sum_U_re += fmy_U_re * cos_ang - fmy_U_im * sin_ang;
                    sum_U_im += fmy_U_re * sin_ang + fmy_U_im * cos_ang;
                }

                map_Q_t[south_ring * max_pix + j] = T(sum_Q_re);
                map_U_t[south_ring * max_pix + j] = T(sum_U_re);
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Spin-2 Bluestein kernels for alm2map
// These handle Q and U components together, using north/south Fmy directly
// ============================================================================

template<typename R>
__global__ void bluestein_spin2_pre_chirp_kernel(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const R* __restrict__ Fmy_Q_north_re,  // [n_maps, lp1, n_north_rings]
    const R* __restrict__ Fmy_Q_north_im,
    const R* __restrict__ Fmy_Q_south_re,
    const R* __restrict__ Fmy_Q_south_im,
    const R* __restrict__ Fmy_U_north_re,
    const R* __restrict__ Fmy_U_north_im,
    const R* __restrict__ Fmy_U_south_re,
    const R* __restrict__ Fmy_U_south_im,
    cufftDoubleComplex* __restrict__ chirped_Q,  // [n_maps, n_rings, M]
    cufftDoubleComplex* __restrict__ chirped_U,
    int* __restrict__ ring_sizes_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    bool is_south = (ring_idx >= n_north_rings);
    int north_ring = is_south ? (n_rings - 1 - ring_idx) : ring_idx;

    // Compute ring geometry
    double phi0;
    int N;
    {
        int ring_i = ring_idx + 1;
        if (ring_i < nside) {
            phi0 = M_PI / (2.0 * ring_i) * 0.5;
            N = 4 * ring_i;
        } else if (ring_i > 3 * nside) {
            int mirror_i = 4 * nside - ring_i;
            phi0 = M_PI / (2.0 * mirror_i) * 0.5;
            N = 4 * mirror_i;
        } else {
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0 = M_PI / (2.0 * nside) * (1.0 - s / 2.0);
            N = 4 * nside;
        }
    }

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    const R* fmy_Q_re = is_south ? Fmy_Q_south_re : Fmy_Q_north_re;
    const R* fmy_Q_im = is_south ? Fmy_Q_south_im : Fmy_Q_north_im;
    const R* fmy_U_re = is_south ? Fmy_U_south_re : Fmy_U_north_re;
    const R* fmy_U_im = is_south ? Fmy_U_south_im : Fmy_U_north_im;

    cufftDoubleComplex* chirped_Q_ring = chirped_Q + (size_t)map_idx * n_rings * M + ring_idx * M;
    cufftDoubleComplex* chirped_U_ring = chirped_U + (size_t)map_idx * n_rings * M + ring_idx * M;
    double pi_over_N = M_PI / double(N);

    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        cufftDoubleComplex val_Q, val_U;
        if (m <= l_max) {
            size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            double fQ_r = double(fmy_Q_re[idx]);
            double fQ_i = double(fmy_Q_im[idx]);
            double fU_r = double(fmy_U_re[idx]);
            double fU_i = double(fmy_U_im[idx]);

            double phase_angle = double(m) * phi0;
            double phase_c, phase_s;
            sincos(phase_angle, &phase_s, &phase_c);
            double fQ_r_corr = fQ_r * phase_c - fQ_i * phase_s;
            double fQ_i_corr = fQ_r * phase_s + fQ_i * phase_c;
            double fU_r_corr = fU_r * phase_c - fU_i * phase_s;
            double fU_i_corr = fU_r * phase_s + fU_i * phase_c;

            double chirp_angle = pi_over_N * double(m) * double(m);
            double chirp_c, chirp_s;
            sincos(chirp_angle, &chirp_s, &chirp_c);

            val_Q.x = fQ_r_corr * chirp_c - fQ_i_corr * chirp_s;
            val_Q.y = fQ_r_corr * chirp_s + fQ_i_corr * chirp_c;
            val_U.x = fU_r_corr * chirp_c - fU_i_corr * chirp_s;
            val_U.y = fU_r_corr * chirp_s + fU_i_corr * chirp_c;
        } else {
            val_Q.x = val_Q.y = val_U.x = val_U.y = 0.0;
        }
        chirped_Q_ring[m] = val_Q;
        chirped_U_ring[m] = val_U;
    }
}

// Float32 version
template<typename R>
__global__ void bluestein_spin2_pre_chirp_kernel_f32(
    int nside, int n_maps, int n_rings, int n_north_rings, int l_max, int M,
    const R* __restrict__ Fmy_Q_north_re,
    const R* __restrict__ Fmy_Q_north_im,
    const R* __restrict__ Fmy_Q_south_re,
    const R* __restrict__ Fmy_Q_south_im,
    const R* __restrict__ Fmy_U_north_re,
    const R* __restrict__ Fmy_U_north_im,
    const R* __restrict__ Fmy_U_south_re,
    const R* __restrict__ Fmy_U_south_im,
    cufftComplex* __restrict__ chirped_Q,
    cufftComplex* __restrict__ chirped_U,
    int* __restrict__ ring_sizes_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    bool is_south = (ring_idx >= n_north_rings);
    int north_ring = is_south ? (n_rings - 1 - ring_idx) : ring_idx;

    float phi0;
    int N;
    {
        int ring_i = ring_idx + 1;
        if (ring_i < nside) {
            phi0 = float(M_PI) / float(2.0f * ring_i) * 0.5f;
            N = 4 * ring_i;
        } else if (ring_i > 3 * nside) {
            int mirror_i = 4 * nside - ring_i;
            phi0 = float(M_PI) / float(2.0f * mirror_i) * 0.5f;
            N = 4 * mirror_i;
        } else {
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0 = float(M_PI) / float(2.0f * nside) * (1.0f - s / 2.0f);
            N = 4 * nside;
        }
    }

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    const R* fmy_Q_re = is_south ? Fmy_Q_south_re : Fmy_Q_north_re;
    const R* fmy_Q_im = is_south ? Fmy_Q_south_im : Fmy_Q_north_im;
    const R* fmy_U_re = is_south ? Fmy_U_south_re : Fmy_U_north_re;
    const R* fmy_U_im = is_south ? Fmy_U_south_im : Fmy_U_north_im;

    cufftComplex* chirped_Q_ring = chirped_Q + (size_t)map_idx * n_rings * M + ring_idx * M;
    cufftComplex* chirped_U_ring = chirped_U + (size_t)map_idx * n_rings * M + ring_idx * M;
    float pi_over_N = float(M_PI) / float(N);

    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        cufftComplex val_Q, val_U;
        if (m <= l_max) {
            size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            float fQ_r = float(fmy_Q_re[idx]);
            float fQ_i = float(fmy_Q_im[idx]);
            float fU_r = float(fmy_U_re[idx]);
            float fU_i = float(fmy_U_im[idx]);

            float phase_angle = float(m) * phi0;
            float phase_c, phase_s;
            sincosf(phase_angle, &phase_s, &phase_c);
            float fQ_r_corr = fQ_r * phase_c - fQ_i * phase_s;
            float fQ_i_corr = fQ_r * phase_s + fQ_i * phase_c;
            float fU_r_corr = fU_r * phase_c - fU_i * phase_s;
            float fU_i_corr = fU_r * phase_s + fU_i * phase_c;

            float chirp_angle = pi_over_N * float(m) * float(m);
            float chirp_c, chirp_s;
            sincosf(chirp_angle, &chirp_s, &chirp_c);

            val_Q.x = fQ_r_corr * chirp_c - fQ_i_corr * chirp_s;
            val_Q.y = fQ_r_corr * chirp_s + fQ_i_corr * chirp_c;
            val_U.x = fU_r_corr * chirp_c - fU_i_corr * chirp_s;
            val_U.y = fU_r_corr * chirp_s + fU_i_corr * chirp_c;
        } else {
            val_Q.x = val_Q.y = val_U.x = val_U.y = 0.0f;
        }
        chirped_Q_ring[m] = val_Q;
        chirped_U_ring[m] = val_U;
    }
}

// Spin-2 extract kernel
template<typename T>
__global__ void bluestein_spin2_extract_kernel(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_Q,
    const cufftDoubleComplex* __restrict__ ifft_U,
    T* __restrict__ map_Q_out,
    T* __restrict__ map_U_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    double pi_over_N = M_PI / double(N);
    double inv_M = 1.0 / double(M);

    const cufftDoubleComplex* ifft_Q_ring = ifft_Q +
        (size_t)map_idx * n_rings * M + ring_idx * M;
    const cufftDoubleComplex* ifft_U_ring = ifft_U +
        (size_t)map_idx * n_rings * M + ring_idx * M;
    T* map_Q_ring = map_Q_out + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;
    T* map_U_ring = map_U_out + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        cufftDoubleComplex zQ = ifft_Q_ring[n];
        cufftDoubleComplex zU = ifft_U_ring[n];
        zQ.x *= inv_M; zQ.y *= inv_M;
        zU.x *= inv_M; zU.y *= inv_M;

        double post_angle = pi_over_N * double(n) * double(n);
        double post_c, post_s;
        sincos(post_angle, &post_s, &post_c);

        double result_Q = zQ.x * post_c - zQ.y * post_s;
        double result_U = zU.x * post_c - zU.y * post_s;

        map_Q_ring[n] = T(result_Q);
        map_U_ring[n] = T(result_U);
    }
}

// Float32 version
template<typename T>
__global__ void bluestein_spin2_extract_kernel_f32(
    int nside, int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_Q,
    const cufftComplex* __restrict__ ifft_U,
    T* __restrict__ map_Q_out,
    T* __restrict__ map_U_out
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    float pi_over_N = float(M_PI) / float(N);
    float inv_M = 1.0f / float(M);

    const cufftComplex* ifft_Q_ring = ifft_Q +
        (size_t)map_idx * n_rings * M + ring_idx * M;
    const cufftComplex* ifft_U_ring = ifft_U +
        (size_t)map_idx * n_rings * M + ring_idx * M;
    T* map_Q_ring = map_Q_out + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;
    T* map_U_ring = map_U_out + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        cufftComplex zQ = ifft_Q_ring[n];
        cufftComplex zU = ifft_U_ring[n];
        zQ.x *= inv_M; zQ.y *= inv_M;
        zU.x *= inv_M; zU.y *= inv_M;

        float post_angle = pi_over_N * float(n) * float(n);
        float post_c, post_s;
        sincosf(post_angle, &post_s, &post_c);

        float result_Q = zQ.x * post_c - zQ.y * post_s;
        float result_U = zU.x * post_c - zU.y * post_s;

        map_Q_ring[n] = T(result_Q);
        map_U_ring[n] = T(result_U);
    }
}

// ============================================================================
// Spin-2 host wrapper
// ============================================================================

template<typename T, typename R>
void alm2map_cuda_v6_spin2_impl(
    int nside, int l_max, int n_maps,
    const T* alm_E_re, const T* alm_E_im,
    const T* alm_B_re, const T* alm_B_im,
    T* map_Q_out, T* map_U_out
) {
    int n_rings = 4 * nside - 1;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;
    int max_pix = 4 * nside;

    int rings_per_lane = (n_north_rings + 31) / 32;
    if (rings_per_lane > MAX_RINGS_PER_LANE) {
        fprintf(stderr, "Error: nside=%d requires %d rings per lane, max is %d\n",
                nside, rings_per_lane, MAX_RINGS_PER_LANE);
        return;
    }

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

    // Allocate scaled alm (m>0 multiplied by 2) for both E and B
    T *alm_E_scaled_re, *alm_E_scaled_im;
    T *alm_B_scaled_re, *alm_B_scaled_im;
    size_t alm_size = (size_t)n_maps * lp1 * lp1 * sizeof(T);

    CUDA_CHECK(cudaMalloc(&alm_E_scaled_re, alm_size));
    CUDA_CHECK(cudaMalloc(&alm_E_scaled_im, alm_size));
    CUDA_CHECK(cudaMalloc(&alm_B_scaled_re, alm_size));
    CUDA_CHECK(cudaMalloc(&alm_B_scaled_im, alm_size));

    dim3 block_scale(16, 16);
    dim3 grid_scale(n_maps, CEILDIV(lp1, 16), CEILDIV(lp1, 16));
    // Use spin-2 scaling kernel that includes norm(l) pre-scaling
    scale_alm_spin2_for_synth_kernel<T><<<grid_scale, block_scale>>>(
        n_maps, lp1, alm_E_re, alm_E_im, alm_E_scaled_re, alm_E_scaled_im
    );
    scale_alm_spin2_for_synth_kernel<T><<<grid_scale, block_scale>>>(
        n_maps, lp1, alm_B_re, alm_B_im, alm_B_scaled_re, alm_B_scaled_im
    );
    CUDA_CHECK(cudaGetLastError());

    if (timing_enabled) {
        cudaEventRecord(end_scale);
        cudaEventRecord(start_p1);
    }

    // Allocate Fmy buffers
    size_t fmy_size = (size_t)n_maps * lp1 * n_north_rings * sizeof(R);
    size_t geom_size = n_north_rings * sizeof(R);

    R *Fmy_Q_north_re, *Fmy_Q_north_im, *Fmy_Q_south_re, *Fmy_Q_south_im;
    R *Fmy_U_north_re, *Fmy_U_north_im, *Fmy_U_south_re, *Fmy_U_south_im;
    R *cos_theta, *sin_theta;

    CUDA_CHECK(cudaMalloc(&Fmy_Q_north_re, fmy_size));
    CUDA_CHECK(cudaMalloc(&Fmy_Q_north_im, fmy_size));
    CUDA_CHECK(cudaMalloc(&Fmy_Q_south_re, fmy_size));
    CUDA_CHECK(cudaMalloc(&Fmy_Q_south_im, fmy_size));
    CUDA_CHECK(cudaMalloc(&Fmy_U_north_re, fmy_size));
    CUDA_CHECK(cudaMalloc(&Fmy_U_north_im, fmy_size));
    CUDA_CHECK(cudaMalloc(&Fmy_U_south_re, fmy_size));
    CUDA_CHECK(cudaMalloc(&Fmy_U_south_im, fmy_size));
    CUDA_CHECK(cudaMalloc(&cos_theta, geom_size));
    CUDA_CHECK(cudaMalloc(&sin_theta, geom_size));

    // Phase 1: Compute Fmy
    // Compute shared memory requirements and adjust ring_batch_size if needed
    const size_t MAX_SMEM = 48 * 1024;  // 48KB default limit
    size_t alm_smem = 4 * lp1 * sizeof(T);  // E/B re/im for all l
    size_t avail_for_rings = (alm_smem < MAX_SMEM) ? (MAX_SMEM - alm_smem) : 0;
    int ring_batch_size = RING_BATCH_SIZE;

    // Reduce ring_batch_size if needed to fit in shared memory
    while (ring_batch_size > 32 && 2 * ring_batch_size * sizeof(R) > avail_for_rings) {
        ring_batch_size /= 2;
    }

    size_t smem_p1 = 2 * ring_batch_size * sizeof(R) + alm_smem;

    // Request extended shared memory if needed (up to 100KB on compute 8.x)
    if (smem_p1 > MAX_SMEM) {
        cudaError_t attr_err = cudaFuncSetAttribute(
            compute_fmy_spin2_kernel_v6<T, R>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_p1);
        if (attr_err != cudaSuccess) {
            fprintf(stderr, "Error: Spin-2 alm2map requires %zu bytes shared memory "
                    "(l_max=%d), but GPU limit exceeded.\n"
                    "Try using float32 storage precision for large nside.\n",
                    smem_p1, l_max);
            return;
        }
    }

    compute_fmy_spin2_kernel_v6<T, R><<<lp1, 32, smem_p1>>>(
        nside, l_max, n_maps, n_north_rings,
        ring_batch_size,
        alm_E_scaled_re, alm_E_scaled_im,
        alm_B_scaled_re, alm_B_scaled_im,
        Fmy_Q_north_re, Fmy_Q_north_im, Fmy_Q_south_re, Fmy_Q_south_im,
        Fmy_U_north_re, Fmy_U_north_im, Fmy_U_south_re, Fmy_U_south_im,
        cos_theta, sin_theta
    );
    CUDA_CHECK(cudaGetLastError());

    if (timing_enabled) {
        cudaEventRecord(end_p1);
        cudaEventRecord(start_p2);
    }

    // Initialize output maps to zero
    CUDA_CHECK(cudaMemset(map_Q_out, 0, (size_t)n_maps * n_rings * max_pix * sizeof(T)));
    CUDA_CHECK(cudaMemset(map_U_out, 0, (size_t)n_maps * n_rings * max_pix * sizeof(T)));

    // Phase 2: Synthesize maps
    // Method selected by g_phase1_method (DFT or Bluestein)

    if (g_phase1_method == Phase1Method::BLUESTEIN) {
        // ============================================================
        // BLUESTEIN INVERSE FFT for Spin-2: Fmy -> Q,U maps
        // ============================================================

        // Compute M = next power of 2 >= l_max + 1 + max_ring_size - 1
        // This is required because Bluestein convolution needs M >= input_length + output_length - 1
        // where input_length = l_max + 1 (number of m values) and output_length = ring_size
        int max_ring_size = 4 * nside;
        int M = next_power_of_2(lp1 + max_ring_size - 1);

        int* ring_sizes;
        CUDA_CHECK(cudaMalloc(&ring_sizes, n_rings * sizeof(int)));

        bool use_double = std::is_same<R, double>::value;

        if (use_double) {
            cufftDoubleComplex* chirped_Q;
            cufftDoubleComplex* chirped_U;
            cufftDoubleComplex* conj_chirp_fft;

            size_t chirp_data_size = (size_t)n_maps * n_rings * M * sizeof(cufftDoubleComplex);
            size_t conj_chirp_size = (size_t)nside * M * sizeof(cufftDoubleComplex);

            CUDA_CHECK(cudaMalloc(&chirped_Q, chirp_data_size));
            CUDA_CHECK(cudaMalloc(&chirped_U, chirp_data_size));
            CUDA_CHECK(cudaMalloc(&conj_chirp_fft, conj_chirp_size));

            // Compute conjugate chirp
            bluestein_compute_inv_chirp_kernel<<<nside, 256>>>(nside, l_max, M, conj_chirp_fft);
            CUDA_CHECK(cudaGetLastError());

            cufftHandle chirp_fft_plan = get_cached_fft_plan(M, nside, CUFFT_Z2Z);
            cufftExecZ2Z(chirp_fft_plan, conj_chirp_fft, conj_chirp_fft, CUFFT_FORWARD);

            // Pre-chirp for Q and U
            dim3 grid_all(n_rings, n_maps);
            bluestein_spin2_pre_chirp_kernel<R><<<grid_all, 256>>>(
                nside, n_maps, n_rings, n_north_rings, l_max, M,
                Fmy_Q_north_re, Fmy_Q_north_im, Fmy_Q_south_re, Fmy_Q_south_im,
                Fmy_U_north_re, Fmy_U_north_im, Fmy_U_south_re, Fmy_U_south_im,
                chirped_Q, chirped_U, ring_sizes
            );
            CUDA_CHECK(cudaGetLastError());

            // FFT both Q and U
            cufftHandle data_fft_plan = get_cached_fft_plan(M, n_maps * n_rings, CUFFT_Z2Z);
            cufftExecZ2Z(data_fft_plan, chirped_Q, chirped_Q, CUFFT_FORWARD);
            cufftExecZ2Z(data_fft_plan, chirped_U, chirped_U, CUFFT_FORWARD);

            // Pointwise multiply
            bluestein_pointwise_mult_kernel_v2<<<grid_all, 256>>>(
                n_maps, n_rings, M, ring_sizes, chirped_Q, conj_chirp_fft
            );
            bluestein_pointwise_mult_kernel_v2<<<grid_all, 256>>>(
                n_maps, n_rings, M, ring_sizes, chirped_U, conj_chirp_fft
            );
            CUDA_CHECK(cudaGetLastError());

            // IFFT both
            cufftExecZ2Z(data_fft_plan, chirped_Q, chirped_Q, CUFFT_INVERSE);
            cufftExecZ2Z(data_fft_plan, chirped_U, chirped_U, CUFFT_INVERSE);

            // Extract Q and U maps
            bluestein_spin2_extract_kernel<T><<<grid_all, 256>>>(
                nside, n_maps, n_rings, M,
                ring_sizes, chirped_Q, chirped_U, map_Q_out, map_U_out
            );
            CUDA_CHECK(cudaGetLastError());

            cudaFree(chirped_Q);
            cudaFree(chirped_U);
            cudaFree(conj_chirp_fft);
        } else {
            // Float32 precision
            cufftComplex* chirped_Q;
            cufftComplex* chirped_U;
            cufftComplex* conj_chirp_fft;

            size_t chirp_data_size = (size_t)n_maps * n_rings * M * sizeof(cufftComplex);
            size_t conj_chirp_size = (size_t)nside * M * sizeof(cufftComplex);

            CUDA_CHECK(cudaMalloc(&chirped_Q, chirp_data_size));
            CUDA_CHECK(cudaMalloc(&chirped_U, chirp_data_size));
            CUDA_CHECK(cudaMalloc(&conj_chirp_fft, conj_chirp_size));

            bluestein_compute_inv_chirp_kernel_f32<<<nside, 256>>>(nside, l_max, M, conj_chirp_fft);
            CUDA_CHECK(cudaGetLastError());

            cufftHandle chirp_fft_plan = get_cached_fft_plan(M, nside, CUFFT_C2C);
            cufftExecC2C(chirp_fft_plan, conj_chirp_fft, conj_chirp_fft, CUFFT_FORWARD);

            dim3 grid_all(n_rings, n_maps);
            bluestein_spin2_pre_chirp_kernel_f32<R><<<grid_all, 256>>>(
                nside, n_maps, n_rings, n_north_rings, l_max, M,
                Fmy_Q_north_re, Fmy_Q_north_im, Fmy_Q_south_re, Fmy_Q_south_im,
                Fmy_U_north_re, Fmy_U_north_im, Fmy_U_south_re, Fmy_U_south_im,
                chirped_Q, chirped_U, ring_sizes
            );
            CUDA_CHECK(cudaGetLastError());

            cufftHandle data_fft_plan = get_cached_fft_plan(M, n_maps * n_rings, CUFFT_C2C);
            cufftExecC2C(data_fft_plan, chirped_Q, chirped_Q, CUFFT_FORWARD);
            cufftExecC2C(data_fft_plan, chirped_U, chirped_U, CUFFT_FORWARD);

            bluestein_pointwise_mult_kernel_f32_v2<<<grid_all, 256>>>(
                n_maps, n_rings, M, ring_sizes, chirped_Q, conj_chirp_fft
            );
            bluestein_pointwise_mult_kernel_f32_v2<<<grid_all, 256>>>(
                n_maps, n_rings, M, ring_sizes, chirped_U, conj_chirp_fft
            );
            CUDA_CHECK(cudaGetLastError());

            cufftExecC2C(data_fft_plan, chirped_Q, chirped_Q, CUFFT_INVERSE);
            cufftExecC2C(data_fft_plan, chirped_U, chirped_U, CUFFT_INVERSE);

            bluestein_spin2_extract_kernel_f32<T><<<grid_all, 256>>>(
                nside, n_maps, n_rings, M,
                ring_sizes, chirped_Q, chirped_U, map_Q_out, map_U_out
            );
            CUDA_CHECK(cudaGetLastError());

            cudaFree(chirped_Q);
            cudaFree(chirped_U);
            cudaFree(conj_chirp_fft);
        }

        cudaFree(ring_sizes);

    } else {
        // ============================================================
        // DIRECT DFT: Default method
        // ============================================================
        size_t smem_p2 = 4 * lp1 * sizeof(R);
        int block_size_p2 = 256;

        // Request extended shared memory if needed
        if (smem_p2 > MAX_SMEM) {
            cudaError_t attr_err = cudaFuncSetAttribute(
                synthesize_map_spin2_kernel_v6<T, R>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_p2);
            if (attr_err != cudaSuccess) {
                fprintf(stderr, "Error: Spin-2 alm2map Phase 2 requires %zu bytes shared memory "
                        "(l_max=%d), but GPU limit exceeded.\n"
                        "Try using float32 storage precision for large nside.\n",
                        smem_p2, l_max);
                // Clean up
                cudaFree(alm_E_scaled_re); cudaFree(alm_E_scaled_im);
                cudaFree(alm_B_scaled_re); cudaFree(alm_B_scaled_im);
                cudaFree(Fmy_Q_north_re); cudaFree(Fmy_Q_north_im);
                cudaFree(Fmy_Q_south_re); cudaFree(Fmy_Q_south_im);
                cudaFree(Fmy_U_north_re); cudaFree(Fmy_U_north_im);
                cudaFree(Fmy_U_south_re); cudaFree(Fmy_U_south_im);
                cudaFree(cos_theta); cudaFree(sin_theta);
                return;
            }
        }

        synthesize_map_spin2_kernel_v6<T, R><<<n_north_rings, block_size_p2, smem_p2>>>(
            nside, l_max, n_maps, n_rings, n_north_rings,
            Fmy_Q_north_re, Fmy_Q_north_im, Fmy_Q_south_re, Fmy_Q_south_im,
            Fmy_U_north_re, Fmy_U_north_im, Fmy_U_south_re, Fmy_U_south_im,
            map_Q_out, map_U_out
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (timing_enabled) {
        cudaEventRecord(end_p2);
        cudaEventSynchronize(end_p2);

        float scale_ms, p1_ms, p2_ms;
        cudaEventElapsedTime(&scale_ms, start_scale, end_scale);
        cudaEventElapsedTime(&p1_ms, start_p1, end_p1);
        cudaEventElapsedTime(&p2_ms, start_p2, end_p2);

        fprintf(stderr, "[SPHT_TIMING] spin2 alm2map nside=%d l_max=%d n_maps=%d Scale=%.2fms Phase1=%.2fms Phase2=%.2fms Total=%.2fms\n",
                nside, l_max, n_maps, scale_ms, p1_ms, p2_ms, scale_ms + p1_ms + p2_ms);

        cudaEventDestroy(start_scale);
        cudaEventDestroy(end_scale);
        cudaEventDestroy(start_p1);
        cudaEventDestroy(end_p1);
        cudaEventDestroy(start_p2);
        cudaEventDestroy(end_p2);
    }

    // Cleanup
    cudaFree(alm_E_scaled_re);
    cudaFree(alm_E_scaled_im);
    cudaFree(alm_B_scaled_re);
    cudaFree(alm_B_scaled_im);
    cudaFree(Fmy_Q_north_re);
    cudaFree(Fmy_Q_north_im);
    cudaFree(Fmy_Q_south_re);
    cudaFree(Fmy_Q_south_im);
    cudaFree(Fmy_U_north_re);
    cudaFree(Fmy_U_north_im);
    cudaFree(Fmy_U_south_re);
    cudaFree(Fmy_U_south_im);
    cudaFree(cos_theta);
    cudaFree(sin_theta);
}

// ============================================================================
// Spin-2 C API entry points
// ============================================================================

extern "C" {

void alm2map_cuda_v6_spin2_f64_f64(int nside, int l_max, int n_maps,
                                    const double* alm_E_re, const double* alm_E_im,
                                    const double* alm_B_re, const double* alm_B_im,
                                    double* map_Q_out, double* map_U_out) {
    alm2map_cuda_v6_spin2_impl<double, double>(nside, l_max, n_maps,
                                                alm_E_re, alm_E_im,
                                                alm_B_re, alm_B_im,
                                                map_Q_out, map_U_out);
}

void alm2map_cuda_v6_spin2_f64_f32(int nside, int l_max, int n_maps,
                                    const double* alm_E_re, const double* alm_E_im,
                                    const double* alm_B_re, const double* alm_B_im,
                                    double* map_Q_out, double* map_U_out) {
    alm2map_cuda_v6_spin2_impl<double, float>(nside, l_max, n_maps,
                                               alm_E_re, alm_E_im,
                                               alm_B_re, alm_B_im,
                                               map_Q_out, map_U_out);
}

void alm2map_cuda_v6_spin2_f32_f64(int nside, int l_max, int n_maps,
                                    const float* alm_E_re, const float* alm_E_im,
                                    const float* alm_B_re, const float* alm_B_im,
                                    float* map_Q_out, float* map_U_out) {
    alm2map_cuda_v6_spin2_impl<float, double>(nside, l_max, n_maps,
                                               alm_E_re, alm_E_im,
                                               alm_B_re, alm_B_im,
                                               map_Q_out, map_U_out);
}

void alm2map_cuda_v6_spin2_f32_f32(int nside, int l_max, int n_maps,
                                    const float* alm_E_re, const float* alm_E_im,
                                    const float* alm_B_re, const float* alm_B_im,
                                    float* map_Q_out, float* map_U_out) {
    alm2map_cuda_v6_spin2_impl<float, float>(nside, l_max, n_maps,
                                              alm_E_re, alm_E_im,
                                              alm_B_re, alm_B_im,
                                              map_Q_out, map_U_out);
}

// Convenience aliases
void alm2map_cuda_v6_spin2_f64(int nside, int l_max, int n_maps,
                                const double* alm_E_re, const double* alm_E_im,
                                const double* alm_B_re, const double* alm_B_im,
                                double* map_Q_out, double* map_U_out) {
    alm2map_cuda_v6_spin2_f64_f64(nside, l_max, n_maps, alm_E_re, alm_E_im,
                                   alm_B_re, alm_B_im, map_Q_out, map_U_out);
}

void alm2map_cuda_v6_spin2_f32(int nside, int l_max, int n_maps,
                                const float* alm_E_re, const float* alm_E_im,
                                const float* alm_B_re, const float* alm_B_im,
                                float* map_Q_out, float* map_U_out) {
    alm2map_cuda_v6_spin2_f32_f32(nside, l_max, n_maps, alm_E_re, alm_E_im,
                                   alm_B_re, alm_B_im, map_Q_out, map_U_out);
}

} // extern "C" for spin-2
