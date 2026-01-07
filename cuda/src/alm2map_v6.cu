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
#include "../include/spht_transform_config.cuh"
#include "../include/bluestein_fft.h"
#include "../include/log_arithmetic.cuh"
#include <stdio.h>
#include <type_traits>

// Ring batch size for Phase 1 (undef to override default from spht_types.h)
#undef RING_BATCH_SIZE
#define RING_BATCH_SIZE 256

// ============================================================================
// Workspace Cache - Eliminates malloc/free overhead on repeated calls
// ============================================================================

template<typename T, typename R>
struct Alm2mapWorkspace {
    // Buffer pointers
    T* alm_scaled_real = nullptr;
    T* alm_scaled_imag = nullptr;
    R* Fmy_even_re = nullptr;
    R* Fmy_even_im = nullptr;
    R* Fmy_odd_re = nullptr;
    R* Fmy_odd_im = nullptr;
    R* cos_theta = nullptr;
    R* sin_theta = nullptr;
    R* Fmy_north_re = nullptr;
    R* Fmy_north_im = nullptr;
    R* Fmy_south_re = nullptr;
    R* Fmy_south_im = nullptr;
    int* ring_sizes = nullptr;
    void* chirped_data = nullptr;
    void* conj_chirp_fft = nullptr;

    // Cached sizes for reallocation check
    size_t alm_size = 0;
    size_t fmy_size = 0;
    size_t geom_size = 0;
    size_t ring_sizes_size = 0;
    size_t chirped_data_size = 0;
    size_t conj_chirp_size = 0;

    void ensure_size(size_t new_alm, size_t new_fmy, size_t new_geom, size_t new_ring_sizes) {
        // Only reallocate if size increased
        if (new_alm > alm_size) {
            if (alm_scaled_real) cudaFree(alm_scaled_real);
            if (alm_scaled_imag) cudaFree(alm_scaled_imag);
            cudaMalloc(&alm_scaled_real, new_alm);
            cudaMalloc(&alm_scaled_imag, new_alm);
            alm_size = new_alm;
        }
        if (new_fmy > fmy_size) {
            if (Fmy_even_re) cudaFree(Fmy_even_re);
            if (Fmy_even_im) cudaFree(Fmy_even_im);
            if (Fmy_odd_re) cudaFree(Fmy_odd_re);
            if (Fmy_odd_im) cudaFree(Fmy_odd_im);
            if (Fmy_north_re) cudaFree(Fmy_north_re);
            if (Fmy_north_im) cudaFree(Fmy_north_im);
            if (Fmy_south_re) cudaFree(Fmy_south_re);
            if (Fmy_south_im) cudaFree(Fmy_south_im);
            cudaMalloc(&Fmy_even_re, new_fmy);
            cudaMalloc(&Fmy_even_im, new_fmy);
            cudaMalloc(&Fmy_odd_re, new_fmy);
            cudaMalloc(&Fmy_odd_im, new_fmy);
            cudaMalloc(&Fmy_north_re, new_fmy);
            cudaMalloc(&Fmy_north_im, new_fmy);
            cudaMalloc(&Fmy_south_re, new_fmy);
            cudaMalloc(&Fmy_south_im, new_fmy);
            fmy_size = new_fmy;
        }
        if (new_geom > geom_size) {
            if (cos_theta) cudaFree(cos_theta);
            if (sin_theta) cudaFree(sin_theta);
            cudaMalloc(&cos_theta, new_geom);
            cudaMalloc(&sin_theta, new_geom);
            geom_size = new_geom;
        }
        if (new_ring_sizes > ring_sizes_size) {
            if (ring_sizes) cudaFree(ring_sizes);
            cudaMalloc(&ring_sizes, new_ring_sizes);
            ring_sizes_size = new_ring_sizes;
        }
    }

    void ensure_bluestein(size_t new_chirped, size_t new_conj) {
        if (new_chirped > chirped_data_size) {
            if (chirped_data) cudaFree(chirped_data);
            cudaMalloc(&chirped_data, new_chirped);
            chirped_data_size = new_chirped;
        }
        if (new_conj > conj_chirp_size) {
            if (conj_chirp_fft) cudaFree(conj_chirp_fft);
            cudaMalloc(&conj_chirp_fft, new_conj);
            conj_chirp_size = new_conj;
        }
    }

    ~Alm2mapWorkspace() {
        if (alm_scaled_real) cudaFree(alm_scaled_real);
        if (alm_scaled_imag) cudaFree(alm_scaled_imag);
        if (Fmy_even_re) cudaFree(Fmy_even_re);
        if (Fmy_even_im) cudaFree(Fmy_even_im);
        if (Fmy_odd_re) cudaFree(Fmy_odd_re);
        if (Fmy_odd_im) cudaFree(Fmy_odd_im);
        if (cos_theta) cudaFree(cos_theta);
        if (sin_theta) cudaFree(sin_theta);
        if (Fmy_north_re) cudaFree(Fmy_north_re);
        if (Fmy_north_im) cudaFree(Fmy_north_im);
        if (Fmy_south_re) cudaFree(Fmy_south_re);
        if (Fmy_south_im) cudaFree(Fmy_south_im);
        if (ring_sizes) cudaFree(ring_sizes);
        if (chirped_data) cudaFree(chirped_data);
        if (conj_chirp_fft) cudaFree(conj_chirp_fft);
    }
};

// Global workspace instances (one per precision combination)
static Alm2mapWorkspace<double, double> g_ws_f64_f64;
static Alm2mapWorkspace<double, float>  g_ws_f64_f32;
static Alm2mapWorkspace<float, double>  g_ws_f32_f64;
static Alm2mapWorkspace<float, float>   g_ws_f32_f32;

template<typename T, typename R>
Alm2mapWorkspace<T, R>& get_workspace();

template<> Alm2mapWorkspace<double, double>& get_workspace<double, double>() { return g_ws_f64_f64; }
template<> Alm2mapWorkspace<double, float>&  get_workspace<double, float>()  { return g_ws_f64_f32; }
template<> Alm2mapWorkspace<float, double>&  get_workspace<float, double>()  { return g_ws_f32_f64; }
template<> Alm2mapWorkspace<float, float>&   get_workspace<float, float>()   { return g_ws_f32_f32; }

// ============================================================================
// Memory budget for kernel configuration (alm2map)
// ============================================================================

// Target: ~1.5KB per thread to avoid spilling to local memory
static constexpr int MEMORY_BUDGET_BYTES_ALM2MAP = 1536;

// Overhead per thread (loop counters, temporaries, etc.)
static constexpr int OVERHEAD_BYTES_ALM2MAP = 128;

// Configuration for dynamically selecting kernel template size
struct KernelConfigAlm2map {
    int rings_per_lane;     // Template parameter to use (16, 32, 64, or 128)
    int n_ring_passes;      // Number of passes to cover all rings
};

// Calculate optimal kernel configuration for alm2map
template<typename R>
KernelConfigAlm2map calculate_kernel_config_alm2map(int nside) {
    // alm2map per-ring arrays:
    // log_Ylm_prev1, log_Ylm_prev2 (2 × sizeof(R))
    // sign_prev1, sign_prev2 (2 × int8_t)
    // log_sin_th_cached, log_cos_th_cached (2 × sizeof(R))
    // sign_cos_th_cached (1 × int8_t)
    // log_Ymm_cached (1 × sizeof(R))
    // sign_Ymm_cached (1 × int8_t)
    // Total: 6 × sizeof(R) + 4 bytes per ring
    int per_ring = 6 * sizeof(R) + 4;

    int available = MEMORY_BUDGET_BYTES_ALM2MAP - OVERHEAD_BYTES_ALM2MAP;
    int max_rings_per_lane = available / per_ring;

    // Select template size (must be power of 2, pick smallest that fits)
    int template_size;
    if (max_rings_per_lane >= 128) template_size = 128;
    else if (max_rings_per_lane >= 64) template_size = 64;
    else if (max_rings_per_lane >= 32) template_size = 32;
    else template_size = 16;

    // Calculate rings needed and passes required
    int n_north_rings = 2 * nside;
    int rings_needed_per_lane = (n_north_rings + 31) / 32;
    int n_ring_passes = (rings_needed_per_lane + template_size - 1) / template_size;

    KernelConfigAlm2map config;
    config.rings_per_lane = template_size;
    config.n_ring_passes = n_ring_passes;
    return config;
}

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
    static constexpr double LOG_PI_VAL = 1.1447298858494002;  // log(π)
    static constexpr double LOG_2_VAL = 0.6931471805599453;   // log(2)
    static constexpr double LOG_3_VAL = 1.0986122886681098;   // log(3)
    static constexpr double LOG_4_VAL = 1.3862943611198906;   // log(4)

    static __device__ __forceinline__ double sqrt_d(double x) { return sqrt(x); }
    static __device__ __forceinline__ double exp_d(double x) { return exp(x); }
    static __device__ __forceinline__ double log_d(double x) { return log(x); }
    static __device__ __forceinline__ void sincos_d(double x, double* s, double* c) { sincos(x, s, c); }
    static __device__ __forceinline__ double load(const double* p) { return __ldg(p); }
};

template<>
struct V6TraitsSynth<float> {
    using storage_t = float;
    using compute_t = float;
    using complex_storage_t = float2;
    static constexpr float PI_VAL = 3.14159265f;
    static constexpr float LOG_PI_VAL = 1.14472988f;  // logf(π)
    static constexpr float LOG_2_VAL = 0.69314718f;   // logf(2)
    static constexpr float LOG_3_VAL = 1.09861228f;   // logf(3)
    static constexpr float LOG_4_VAL = 1.38629436f;   // logf(4)

    static __device__ __forceinline__ float sqrt_d(float x) { return sqrtf(x); }
    static __device__ __forceinline__ float exp_d(float x) { return expf(x); }
    static __device__ __forceinline__ float log_d(float x) { return logf(x); }
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

    // Precompute log(nside) for log arithmetic
    C log_nside = Traits::log_d(C(nside));

    if (ring_i < nside) {
        // North polar cap
        // i2_3n2 = ring_i² / (3 * nside²) using log arithmetic
        C log_ring_i = Traits::log_d(C(ring_i));
        C i2_3n2 = Traits::exp_d(C(2.0) * log_ring_i - Traits::LOG_3_VAL - C(2.0) * log_nside);
        cos_th = C(1.0) - i2_3n2;
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        // phi0 = PI / (4 * ring_i) using log arithmetic
        phi0 = Traits::exp_d(Traits::LOG_PI_VAL - Traits::LOG_4_VAL - log_ring_i);
        npix = 4 * ring_i;
    } else if (ring_i > 3 * nside) {
        // South polar cap
        int mirror_i = 4 * nside - ring_i;
        C log_mirror_i = Traits::log_d(C(mirror_i));
        C i2_3n2 = Traits::exp_d(C(2.0) * log_mirror_i - Traits::LOG_3_VAL - C(2.0) * log_nside);
        cos_th = -(C(1.0) - i2_3n2);
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        phi0 = Traits::exp_d(Traits::LOG_PI_VAL - Traits::LOG_4_VAL - log_mirror_i);
        npix = 4 * mirror_i;
    } else {
        // Equatorial belt
        // cos_th = 4/3 - 2*ring_i/(3*nside), use log for the division part
        C log_ring_i = Traits::log_d(C(ring_i));
        C term = Traits::exp_d(Traits::LOG_2_VAL + log_ring_i - Traits::LOG_3_VAL - log_nside);
        cos_th = C(4.0 / 3.0) - term;
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        int s = (ring_i % 2 == 0) ? 1 : 2;
        // phi0 = PI / (2 * nside) * (1 - s/2)
        phi0 = Traits::exp_d(Traits::LOG_PI_VAL - Traits::LOG_2_VAL - log_nside) * C(1.0 - s / 2.0);
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

template<typename T, typename R, bool USE_PRECOMPUTED_COEFF, int RINGS_PER_LANE>
__global__ void compute_fmy_kernel_v6(
    int nside, int l_max, int n_maps, int n_north_rings,
    int ring_batch_size,
    int ring_pass, int total_ring_passes,  // Multi-pass ring processing
    int tile_size,  // L-tiling: 0 means no tiling (use lp1), >0 means tile size
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

    // Effective tile size: 0 means full lp1 (no tiling)
    int eff_tile_size = (tile_size > 0) ? tile_size : lp1;

    // Precompute m-dependent values (used throughout kernel)
    C m2_precomp = C(m * m);
    C recur_c_m1 = Traits::sqrt_d(C(2*m + 3));  // For l = m+1 recurrence

    // Shared memory layout - alm arrays use tile_size when tiling
    extern __shared__ char smem[];
    R* sh_cos_th = (R*)smem;
    R* sh_sin_th = sh_cos_th + ring_batch_size;
    T* sh_alm_re = (T*)(sh_sin_th + ring_batch_size);
    T* sh_alm_im = sh_alm_re + eff_tile_size;
    // Precomputed recurrence coefficients (only if USE_PRECOMPUTED_COEFF and not tiling)
    // When tiling, we always compute coefficients on the fly
    C* sh_log_A = (USE_PRECOMPUTED_COEFF && tile_size == 0) ? (C*)(sh_alm_im + eff_tile_size) : nullptr;
    C* sh_log_B = (USE_PRECOMPUTED_COEFF && tile_size == 0) ? (sh_log_A + lp1) : nullptr;

    // Per-lane Ylm recurrence state in LOG-SPACE
    // We store log(|Ylm|) and sign separately for numerical stability
    // Array sizes determined by template parameter RINGS_PER_LANE
    C log_Ylm_prev1[RINGS_PER_LANE];
    C log_Ylm_prev2[RINGS_PER_LANE];
    int8_t sign_prev1[RINGS_PER_LANE];
    int8_t sign_prev2[RINGS_PER_LANE];
    // Cached per-ring values to avoid recomputing for each map
    C log_sin_th_cached[RINGS_PER_LANE];
    C log_cos_th_cached[RINGS_PER_LANE];
    int8_t sign_cos_th_cached[RINGS_PER_LANE];
    C log_Ymm_cached[RINGS_PER_LANE];
    int8_t sign_Ymm_cached[RINGS_PER_LANE];

    // Log-space traits
    using LogTraits = LogArithmeticTraits<C>;

    // Precompute m-dependent constants ONCE (not per batch)
    const C log_prefact = compute_log_prefact_ymm<C>(m);
    const C log_norm = -C(0.5) * LogTraits::log_d(C(4.0) * LogTraits::PI_VAL);
    const C log_recur_C_m1 = C(0.5) * LogTraits::log_d(C(2*m + 3));  // For l = m+1

    // Multi-pass ring processing: compute ring range for this pass
    // k ranges from 0 to n_my_rings_per_pass-1, where each k corresponds to global ring lane + 32*k
    int n_my_rings_total = (n_north_rings + 31 - lane) / 32;
    int k_offset = ring_pass * RINGS_PER_LANE;
    int k_max_this_pass = min(n_my_rings_total - k_offset, RINGS_PER_LANE);

    // Precompute recurrence coefficients for this m (only if enabled and not tiling)
    if (USE_PRECOMPUTED_COEFF && tile_size == 0) {
        for (int l = m + 2 + lane; l <= l_max; l += 32) {
            sh_log_A[l] = log_A_lm<C>(l, m);
            sh_log_B[l] = log_B_lm<C>(l, m);
        }
        __syncwarp();
    }

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

        // Determine which rings this lane handles in this batch (accounting for ring_pass)
        // k values for this pass range from k_offset to k_offset + k_max_this_pass - 1
        int k_batch_start = (batch_start > lane) ? (batch_start - lane + 31) / 32 : 0;
        int k_batch_end = (batch_end > lane) ? (batch_end - 1 - lane) / 32 + 1 : 0;
        // Intersect batch range with ring_pass range, then shift to 0-based for local arrays
        int k_start = max(k_batch_start, k_offset) - k_offset;
        int k_end = min(k_batch_end, k_offset + k_max_this_pass) - k_offset;
        k_start = max(k_start, 0);
        k_end = max(k_end, 0);

        // Initialize Y[m,m] in LOG-SPACE for rings in this batch
        for (int k = k_start; k < k_end; k++) {
            int global_r = lane + 32 * (k + k_offset);
            int local_r = global_r - batch_start;
            C sin_th = C(sh_sin_th[local_r]);
            C cos_th = C(sh_cos_th[local_r]);

            // Cache per-ring log values (reused across all maps)
            C log_sin_th = safe_log_typed<C>(sin_th);
            C log_cos_th = safe_log_typed<C>(cos_th);
            log_sin_th_cached[k] = log_sin_th;
            log_cos_th_cached[k] = log_cos_th;
            sign_cos_th_cached[k] = (cos_th >= C(0)) ? int8_t(1) : int8_t(-1);

            // Y[m,m] = (-1)^m * sin(th)^m * prefact / sqrt(4*pi)
            C log_Ymm = C(m) * log_sin_th + log_prefact + log_norm;
            int8_t sign_Ymm = ((m & 1) == 0) ? int8_t(1) : int8_t(-1);  // (-1)^m
            log_Ymm_cached[k] = log_Ymm;
            sign_Ymm_cached[k] = sign_Ymm;

            log_Ylm_prev1[k] = log_Ymm;
            sign_prev1[k] = sign_Ymm;
            log_Ylm_prev2[k] = LogTraits::LOG_MIN;  // Zero in log-space
            sign_prev2[k] = 0;
        }

        __syncwarp();

        // Process each map
        for (int t = 0; t < n_maps; t++) {
            const T* alm_re_t = alm_real + (size_t)t * lp1 * lp1;
            const T* alm_im_t = alm_imag + (size_t)t * lp1 * lp1;

            // Initialize per-ring state for tiled processing
            // Fmy accumulators (persist across tiles)
            C fmy_n_re_acc[RINGS_PER_LANE], fmy_n_im_acc[RINGS_PER_LANE];
            C fmy_s_re_acc[RINGS_PER_LANE], fmy_s_im_acc[RINGS_PER_LANE];
            // Ylm recurrence state (persists across tiles)
            C log_Ylm_p1_state[RINGS_PER_LANE], log_Ylm_p2_state[RINGS_PER_LANE];
            int8_t sign_p1_state[RINGS_PER_LANE], sign_p2_state[RINGS_PER_LANE];

            for (int k = k_start; k < k_end; k++) {
                fmy_n_re_acc[k] = C(0); fmy_n_im_acc[k] = C(0);
                fmy_s_re_acc[k] = C(0); fmy_s_im_acc[k] = C(0);
                // Initialize Ylm state from cached Y[m,m]
                log_Ylm_p1_state[k] = log_Ymm_cached[k];
                sign_p1_state[k] = sign_Ymm_cached[k];
                log_Ylm_p2_state[k] = LogTraits::LOG_MIN;
                sign_p2_state[k] = 0;
            }

            // L-tiled processing: iterate over l in tiles
            for (int l_tile = m; l_tile <= l_max; l_tile += eff_tile_size) {
                int l_tile_end = min(l_tile + eff_tile_size, l_max + 1);

                // Cooperative load of alm tile (all threads participate)
                for (int l = l_tile + lane; l < l_tile_end; l += 32) {
                    int local_l = l - l_tile;
                    sh_alm_re[local_l] = alm_re_t[l * lp1 + m];
                    sh_alm_im[local_l] = alm_im_t[l * lp1 + m];
                }
                __syncwarp();

                // Each thread processes its rings for this tile
                for (int k = k_start; k < k_end; k++) {
                    C log_cos_th = log_cos_th_cached[k];
                    int8_t sign_cos_th = sign_cos_th_cached[k];

                    // Restore Ylm state from previous tile
                    C log_Ylm_p1 = log_Ylm_p1_state[k];
                    C log_Ylm_p2 = log_Ylm_p2_state[k];
                    int8_t sign_p1 = sign_p1_state[k];
                    int8_t sign_p2 = sign_p2_state[k];

                    // Local accumulators for this tile (will add to persistent)
                    C fmy_n_re = C(0), fmy_n_im = C(0);
                    C fmy_s_re = C(0), fmy_s_im = C(0);

                    // Process l values in this tile
                    for (int l = l_tile; l < l_tile_end; l++) {
                        int local_l = l - l_tile;
                        C log_Ylm;
                        int8_t sign_Ylm;

                        if (l == m) {
                            log_Ylm = log_Ymm_cached[k];
                            sign_Ylm = sign_Ymm_cached[k];
                        } else if (l == m + 1) {
                            // Y[m+1,m] = cos_th * sqrt(2m+3) * Y[m,m]
                            log_Ylm = log_cos_th + log_recur_C_m1 + log_Ylm_p1;
                            sign_Ylm = sign_cos_th * sign_p1;

                            log_Ylm_p2 = log_Ylm_p1;
                            sign_p2 = sign_p1;
                            log_Ylm_p1 = log_Ylm;
                            sign_p1 = sign_Ylm;
                        } else {
                            // Y[l,m] = A*cos_th*Y[l-1,m] - B*Y[l-2,m]
                            // When tiling, always compute coefficients on the fly
                            C log_A = (USE_PRECOMPUTED_COEFF && tile_size == 0) ? sh_log_A[l] : log_A_lm<C>(l, m);
                            C log_B = (USE_PRECOMPUTED_COEFF && tile_size == 0) ? sh_log_B[l] : log_B_lm<C>(l, m);
                            C R1 = log_A + log_cos_th + log_Ylm_p1;
                            int8_t S1 = sign_cos_th * sign_p1;
                            C R2 = log_B + log_Ylm_p2;
                            int8_t S2 = -sign_p2;  // Negative due to subtraction

                            logsumexp_fast<C>(R1, R2, S1, S2, &log_Ylm, &sign_Ylm);

                            log_Ylm_p2 = log_Ylm_p1;
                            sign_p2 = sign_p1;
                            log_Ylm_p1 = log_Ylm;
                            sign_p1 = sign_Ylm;
                        }

                        // Convert to LINEAR for alm * Ylm multiplication
                        C Ylm = sign_Ylm * LogTraits::exp_d(log_Ylm);

                        // alm * Ylm contribution (use tiled index)
                        C alm_re = C(sh_alm_re[local_l]);
                        C alm_im = C(sh_alm_im[local_l]);

                        // North: Fmy += alm * Ylm
                        fmy_n_re += alm_re * Ylm;
                        fmy_n_im += alm_im * Ylm;

                        // South: Ylm(-cos_theta) = (-1)^(l+m) * Ylm(cos_theta)
                        int parity = (l + m) & 1;
                        C parity_sign = parity ? C(-1) : C(1);
                        fmy_s_re += alm_re * (parity_sign * Ylm);
                        fmy_s_im += alm_im * (parity_sign * Ylm);
                    }

                    // Save Ylm state for next tile
                    log_Ylm_p1_state[k] = log_Ylm_p1;
                    log_Ylm_p2_state[k] = log_Ylm_p2;
                    sign_p1_state[k] = sign_p1;
                    sign_p2_state[k] = sign_p2;

                    // Accumulate to persistent accumulators
                    fmy_n_re_acc[k] += fmy_n_re;
                    fmy_n_im_acc[k] += fmy_n_im;
                    fmy_s_re_acc[k] += fmy_s_re;
                    fmy_s_im_acc[k] += fmy_s_im;
                }
                // No syncwarp needed - single warp in lockstep
            }

            // Store final accumulated results
            for (int k = k_start; k < k_end; k++) {
                int global_r = lane + 32 * (k + k_offset);
                size_t idx = (size_t)t * lp1 * n_north_rings + (size_t)m * n_north_rings + global_r;
                Fmy_even_re[idx] = R(fmy_n_re_acc[k] + fmy_s_re_acc[k]);
                Fmy_even_im[idx] = R(fmy_n_im_acc[k] + fmy_s_im_acc[k]);
                Fmy_odd_re[idx]  = R(fmy_n_re_acc[k] - fmy_s_re_acc[k]);
                Fmy_odd_im[idx]  = R(fmy_n_im_acc[k] - fmy_s_im_acc[k]);
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
                // Use modular arithmetic to avoid precision loss when m is large
                // angle = m*phi0 + m*j*2π/N = m*phi0 + (m*j mod N)*2π/N + k*2π
                // The k*2π term doesn't affect sin/cos, so we use (m*j mod N)
                int mj_mod_N = (m * j) % n_pix_n;
                C angle = C(m) * C(phi0_n) + C(mj_mod_N) * C(2.0 * Traits::PI_VAL) / C(n_pix_n);
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

                    // Use modular arithmetic to avoid precision loss when m is large
                    int mj_mod_N = (m * j) % n_pix_s;
                    C angle = C(m) * C(phi0_s) + C(mj_mod_N) * C(2.0 * Traits::PI_VAL) / C(n_pix_s);
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
        constexpr float LOG_PI = 1.14472988f;
        constexpr float LOG_2 = 0.69314718f;
        constexpr float LOG_3 = 1.09861228f;
        constexpr float LOG_4 = 1.38629436f;
        float log_nside = logf(float(nside));

        int ring_i = ring_idx + 1;
        if (ring_i < nside) {
            float log_ring_i = logf(float(ring_i));
            float i2_3n2 = expf(2.0f * log_ring_i - LOG_3 - 2.0f * log_nside);
            cos_th = 1.0f - i2_3n2;
            sin_th = sqrtf(1.0f - cos_th * cos_th);
            phi0 = expf(LOG_PI - LOG_4 - log_ring_i);
            N = 4 * ring_i;
        } else if (ring_i > 3 * nside) {
            int mirror_i = 4 * nside - ring_i;
            float log_mirror_i = logf(float(mirror_i));
            float i2_3n2 = expf(2.0f * log_mirror_i - LOG_3 - 2.0f * log_nside);
            cos_th = -(1.0f - i2_3n2);
            sin_th = sqrtf(1.0f - cos_th * cos_th);
            phi0 = expf(LOG_PI - LOG_4 - log_mirror_i);
            N = 4 * mirror_i;
        } else {
            float log_ring_i = logf(float(ring_i));
            float term = expf(LOG_2 + log_ring_i - LOG_3 - log_nside);
            cos_th = 4.0f / 3.0f - term;
            sin_th = sqrtf(1.0f - cos_th * cos_th);
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0 = expf(LOG_PI - LOG_2 - log_nside) * (1.0f - s / 2.0f);
            N = 4 * nside;
        }
    }

    if (map_idx == 0 && threadIdx.x == 0) {
        ring_sizes_out[ring_idx] = N;
    }

    const R* fmy_re = is_south ? Fmy_south_re : Fmy_north_re;
    const R* fmy_im = is_south ? Fmy_south_im : Fmy_north_im;

    cufftComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;
    constexpr float LOG_PI = 1.14472988f;  // logf(π)
    float log_N = logf(float(N));

    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        cufftComplex val;
        if (m <= l_max) {
            size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            float fmy_r = float(fmy_re[idx]);
            float fmy_i = float(fmy_im[idx]);

            // phase_angle = m * phi0
            float phase_angle = float(m) * phi0;
            float phase_c, phase_s;
            sincosf(phase_angle, &phase_s, &phase_c);
            float fmy_r_corr = fmy_r * phase_c - fmy_i * phase_s;
            float fmy_i_corr = fmy_r * phase_s + fmy_i * phase_c;

            // Use log arithmetic: chirp_angle = π*m²/N = exp(log(π) + 2*log(m) - log(N))
            float chirp_angle = (m == 0) ? 0.0f : expf(LOG_PI + 2.0f * logf(float(m)) - log_N);
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
    constexpr float LOG_PI = 1.14472988f;  // logf(π)
    float log_N = logf(float(N));
    float inv_M = 1.0f / float(M);

    const cufftComplex* ifft_ring = ifft_data +
        (size_t)map_idx * n_rings * M + ring_idx * M;
    T* map_ring = map_out + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        cufftComplex z = ifft_ring[n];
        z.x *= inv_M;
        z.y *= inv_M;

        // Use log arithmetic: post_angle = π * n² / N = exp(log(π) + 2*log(n) - log(N))
        float post_angle = (n == 0) ? 0.0f : expf(LOG_PI + 2.0f * logf(float(n)) - log_N);
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
        double pi_over_N = M_PI / double(N);
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
    constexpr float LOG_PI = 1.14472988f;  // logf(π)
    float log_N = logf(float(N));

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

        // Use log arithmetic: angle = π * j_eff² / N = exp(log(π) + 2*log(j_eff) - log(N))
        float angle = (j_eff == 0) ? 0.0f : expf(LOG_PI + 2.0f * logf(float(j_eff)) - log_N);
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

    // Calculate optimal kernel configuration
    KernelConfigAlm2map config = calculate_kernel_config_alm2map<R>(nside);

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

    // Get cached workspace (avoids malloc/free overhead on repeated calls)
    Alm2mapWorkspace<T, R>& ws = get_workspace<T, R>();

    // Compute required sizes
    size_t alm_size = (size_t)n_maps * lp1 * lp1 * sizeof(T);
    size_t fmy_size = (size_t)n_maps * lp1 * n_north_rings * sizeof(R);
    size_t geom_size = n_north_rings * sizeof(R);
    size_t ring_sizes_size = n_rings * sizeof(int);

    // Ensure workspace has sufficient size (only reallocates if needed)
    ws.ensure_size(alm_size, fmy_size, geom_size, ring_sizes_size);

    // Use cached pointers
    T* alm_scaled_real = ws.alm_scaled_real;
    T* alm_scaled_imag = ws.alm_scaled_imag;
    R* Fmy_even_re = ws.Fmy_even_re;
    R* Fmy_even_im = ws.Fmy_even_im;
    R* Fmy_odd_re = ws.Fmy_odd_re;
    R* Fmy_odd_im = ws.Fmy_odd_im;
    R* cos_theta = ws.cos_theta;
    R* sin_theta = ws.sin_theta;

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

    // Phase 1: Compute Fmy = sum_l(alm * Ylm)
    // Shared memory layout: geometry (2 arrays) + alm (2 arrays) + optional coefficients (2 arrays)
    // With L-tiling, alm arrays can be smaller than lp1
    const size_t MAX_SMEM = 48 * 1024;  // Default limit, can be extended to ~100KB
    const size_t EXTENDED_SMEM = 100 * 1024;  // Extended limit on compute 8.x

    // Calculate minimum shared memory for geometry (ring batching)
    size_t geom_smem = 2 * RING_BATCH_SIZE * sizeof(R);

    // Calculate alm shared memory if no tiling (full lp1)
    size_t alm_smem_full = 2 * lp1 * sizeof(T);
    size_t coeff_smem_full = 2 * lp1 * sizeof(R);

    // Determine if tiling is needed and compute tile_size
    int tile_size = 0;  // 0 means no tiling (use full lp1)
    int ring_batch_size = RING_BATCH_SIZE;
    size_t smem_p1;
    bool use_precomputed;

    // Try without tiling first
    if (alm_smem_full + coeff_smem_full + geom_smem <= MAX_SMEM) {
        // Everything fits with precomputed coefficients
        use_precomputed = true;
        smem_p1 = geom_smem + alm_smem_full + coeff_smem_full;
    } else if (alm_smem_full + geom_smem <= MAX_SMEM) {
        // Fits without precomputed coefficients
        use_precomputed = false;
        smem_p1 = geom_smem + alm_smem_full;
    } else if (alm_smem_full + geom_smem <= EXTENDED_SMEM) {
        // Need extended shared memory, no precomputed coefficients
        use_precomputed = false;
        smem_p1 = geom_smem + alm_smem_full;
    } else {
        // Need L-tiling: alm doesn't fit even with extended shared memory
        // Compute tile_size to fit within MAX_SMEM (more conservative for occupancy)
        use_precomputed = false;
        size_t avail_for_alm = MAX_SMEM - geom_smem;
        tile_size = (int)(avail_for_alm / (2 * sizeof(T)));
        // Round down to multiple of 32 for coalesced access
        tile_size = (tile_size / 32) * 32;
        tile_size = max(256, tile_size);  // Minimum tile size for efficiency

        size_t alm_smem_tiled = 2 * tile_size * sizeof(T);
        smem_p1 = geom_smem + alm_smem_tiled;
    }

    // Request extended shared memory if needed (up to 100KB on compute 8.x)
    if (smem_p1 > MAX_SMEM) {
        #define SET_SMEM_ATTR(RINGS_PER_LANE_VAL, USE_PRECOMP) \
            cudaFuncSetAttribute(compute_fmy_kernel_v6<T, R, USE_PRECOMP, RINGS_PER_LANE_VAL>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_p1)

        cudaError_t attr_err;
        if (use_precomputed) {
            switch (config.rings_per_lane) {
                case 16:  attr_err = SET_SMEM_ATTR(16, true);  break;
                case 32:  attr_err = SET_SMEM_ATTR(32, true);  break;
                case 64:  attr_err = SET_SMEM_ATTR(64, true);  break;
                case 128: attr_err = SET_SMEM_ATTR(128, true); break;
                default:  attr_err = cudaErrorInvalidValue;    break;
            }
        } else {
            switch (config.rings_per_lane) {
                case 16:  attr_err = SET_SMEM_ATTR(16, false);  break;
                case 32:  attr_err = SET_SMEM_ATTR(32, false);  break;
                case 64:  attr_err = SET_SMEM_ATTR(64, false);  break;
                case 128: attr_err = SET_SMEM_ATTR(128, false); break;
                default:  attr_err = cudaErrorInvalidValue;     break;
            }
        }
        #undef SET_SMEM_ATTR

        if (attr_err != cudaSuccess) {
            fprintf(stderr, "Error: alm2map Phase 1 requires %zu bytes shared memory "
                    "(l_max=%d), but GPU limit exceeded.\n",
                    smem_p1, l_max);
            return;
        }
    }

    // Debug output for transforms (only when timing enabled)
    if (tile_size > 0 && timing_enabled) {
        fprintf(stderr, "alm2map Phase 1: Using L-tiling with tile_size=%d (l_max=%d)\n",
                tile_size, l_max);
    }

    // Dispatch macro for kernel launch with template instantiation
    #define DISPATCH_FMY_KERNEL(RINGS_PER_LANE_VAL, USE_PRECOMP) \
        compute_fmy_kernel_v6<T, R, USE_PRECOMP, RINGS_PER_LANE_VAL><<<lp1, 32, smem_p1>>>( \
            nside, l_max, n_maps, n_north_rings, \
            ring_batch_size, \
            ring_pass, config.n_ring_passes, \
            tile_size, \
            alm_scaled_real, alm_scaled_imag, \
            Fmy_even_re, Fmy_even_im, Fmy_odd_re, Fmy_odd_im, \
            cos_theta, sin_theta \
        )

    // Multi-pass ring processing loop
    for (int ring_pass = 0; ring_pass < config.n_ring_passes; ring_pass++) {
        if (use_precomputed) {
            switch (config.rings_per_lane) {
                case 16:  DISPATCH_FMY_KERNEL(16, true);  break;
                case 32:  DISPATCH_FMY_KERNEL(32, true);  break;
                case 64:  DISPATCH_FMY_KERNEL(64, true);  break;
                case 128: DISPATCH_FMY_KERNEL(128, true); break;
                default:
                    fprintf(stderr, "Error: Unsupported rings_per_lane=%d\n", config.rings_per_lane);
                    return;
            }
        } else {
            switch (config.rings_per_lane) {
                case 16:  DISPATCH_FMY_KERNEL(16, false);  break;
                case 32:  DISPATCH_FMY_KERNEL(32, false);  break;
                case 64:  DISPATCH_FMY_KERNEL(64, false);  break;
                case 128: DISPATCH_FMY_KERNEL(128, false); break;
                default:
                    fprintf(stderr, "Error: Unsupported rings_per_lane=%d\n", config.rings_per_lane);
                    return;
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }

    #undef DISPATCH_FMY_KERNEL

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

        // Use cached Fmy_north/south buffers from workspace
        R* Fmy_north_re = ws.Fmy_north_re;
        R* Fmy_north_im = ws.Fmy_north_im;
        R* Fmy_south_re = ws.Fmy_south_re;
        R* Fmy_south_im = ws.Fmy_south_im;

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

        // Use cached ring_sizes buffer from workspace
        int* ring_sizes = ws.ring_sizes;

        bool use_double = std::is_same<R, double>::value;

        if (use_double) {
            // Double precision Bluestein inverse
            size_t chirp_data_size = (size_t)n_maps * n_rings * M * sizeof(cufftDoubleComplex);
            size_t conj_chirp_size = (size_t)nside * M * sizeof(cufftDoubleComplex);

            // Use cached Bluestein buffers
            ws.ensure_bluestein(chirp_data_size, conj_chirp_size);
            cufftDoubleComplex* chirped_data = (cufftDoubleComplex*)ws.chirped_data;
            cufftDoubleComplex* conj_chirp_fft = (cufftDoubleComplex*)ws.conj_chirp_fft;

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

            // No cudaFree - buffers are cached in workspace
        } else {
            // Float32 precision Bluestein inverse
            size_t chirp_data_size = (size_t)n_maps * n_rings * M * sizeof(cufftComplex);
            size_t conj_chirp_size = (size_t)nside * M * sizeof(cufftComplex);

            // Use cached Bluestein buffers
            ws.ensure_bluestein(chirp_data_size, conj_chirp_size);
            cufftComplex* chirped_data = (cufftComplex*)ws.chirped_data;
            cufftComplex* conj_chirp_fft = (cufftComplex*)ws.conj_chirp_fft;

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

            // No cudaFree - buffers are cached in workspace
        }

        // No cudaFree - all buffers are cached in workspace

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
                // No cleanup needed - workspace is cached
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

    // No cleanup needed - all buffers are cached in workspace for reuse
}

// ============================================================================
// Explicit template instantiations for compute_fmy_kernel_v6
// Each combination: T (storage), R (recurrence), USE_PRECOMPUTED, RINGS_PER_LANE
// ============================================================================

// Double storage, double recurrence
template __global__ void compute_fmy_kernel_v6<double, double, true, 16>(int, int, int, int, int, int, int, int, const double*, const double*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<double, double, true, 32>(int, int, int, int, int, int, int, int, const double*, const double*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<double, double, true, 64>(int, int, int, int, int, int, int, int, const double*, const double*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<double, double, true, 128>(int, int, int, int, int, int, int, int, const double*, const double*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<double, double, false, 16>(int, int, int, int, int, int, int, int, const double*, const double*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<double, double, false, 32>(int, int, int, int, int, int, int, int, const double*, const double*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<double, double, false, 64>(int, int, int, int, int, int, int, int, const double*, const double*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<double, double, false, 128>(int, int, int, int, int, int, int, int, const double*, const double*, double*, double*, double*, double*, double*, double*);

// Double storage, float recurrence
template __global__ void compute_fmy_kernel_v6<double, float, true, 16>(int, int, int, int, int, int, int, int, const double*, const double*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<double, float, true, 32>(int, int, int, int, int, int, int, int, const double*, const double*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<double, float, true, 64>(int, int, int, int, int, int, int, int, const double*, const double*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<double, float, true, 128>(int, int, int, int, int, int, int, int, const double*, const double*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<double, float, false, 16>(int, int, int, int, int, int, int, int, const double*, const double*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<double, float, false, 32>(int, int, int, int, int, int, int, int, const double*, const double*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<double, float, false, 64>(int, int, int, int, int, int, int, int, const double*, const double*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<double, float, false, 128>(int, int, int, int, int, int, int, int, const double*, const double*, float*, float*, float*, float*, float*, float*);

// Float storage, double recurrence
template __global__ void compute_fmy_kernel_v6<float, double, true, 16>(int, int, int, int, int, int, int, int, const float*, const float*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<float, double, true, 32>(int, int, int, int, int, int, int, int, const float*, const float*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<float, double, true, 64>(int, int, int, int, int, int, int, int, const float*, const float*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<float, double, true, 128>(int, int, int, int, int, int, int, int, const float*, const float*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<float, double, false, 16>(int, int, int, int, int, int, int, int, const float*, const float*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<float, double, false, 32>(int, int, int, int, int, int, int, int, const float*, const float*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<float, double, false, 64>(int, int, int, int, int, int, int, int, const float*, const float*, double*, double*, double*, double*, double*, double*);
template __global__ void compute_fmy_kernel_v6<float, double, false, 128>(int, int, int, int, int, int, int, int, const float*, const float*, double*, double*, double*, double*, double*, double*);

// Float storage, float recurrence
template __global__ void compute_fmy_kernel_v6<float, float, true, 16>(int, int, int, int, int, int, int, int, const float*, const float*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<float, float, true, 32>(int, int, int, int, int, int, int, int, const float*, const float*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<float, float, true, 64>(int, int, int, int, int, int, int, int, const float*, const float*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<float, float, true, 128>(int, int, int, int, int, int, int, int, const float*, const float*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<float, float, false, 16>(int, int, int, int, int, int, int, int, const float*, const float*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<float, float, false, 32>(int, int, int, int, int, int, int, int, const float*, const float*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<float, float, false, 64>(int, int, int, int, int, int, int, int, const float*, const float*, float*, float*, float*, float*, float*, float*);
template __global__ void compute_fmy_kernel_v6<float, float, false, 128>(int, int, int, int, int, int, int, int, const float*, const float*, float*, float*, float*, float*, float*, float*);

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

            for (int m = 0; m <= l_max; m++) {
                C fmy_Q_re = C(sh_fmy_Q_re[m]);
                C fmy_Q_im = C(sh_fmy_Q_im[m]);
                C fmy_U_re = C(sh_fmy_U_re[m]);
                C fmy_U_im = C(sh_fmy_U_im[m]);

                // Use modular arithmetic to avoid precision loss when m is large
                int mj_mod_N = (m * j) % n_pix_n;
                C angle = C(m) * C(phi0_n) + C(mj_mod_N) * C(2.0 * Traits::PI_VAL) / C(n_pix_n);
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

                for (int m = 0; m <= l_max; m++) {
                    C fmy_Q_re = C(sh_fmy_Q_re[m]);
                    C fmy_Q_im = C(sh_fmy_Q_im[m]);
                    C fmy_U_re = C(sh_fmy_U_re[m]);
                    C fmy_U_im = C(sh_fmy_U_im[m]);

                    // Use modular arithmetic to avoid precision loss when m is large
                    int mj_mod_N = (m * j) % n_pix_s;
                    C angle = C(m) * C(phi0_s) + C(mj_mod_N) * C(2.0 * Traits::PI_VAL) / C(n_pix_s);
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
        constexpr float LOG_PI = 1.14472988f;
        constexpr float LOG_2 = 0.69314718f;
        constexpr float LOG_4 = 1.38629436f;
        float log_nside = logf(float(nside));

        int ring_i = ring_idx + 1;
        if (ring_i < nside) {
            phi0 = expf(LOG_PI - LOG_4 - logf(float(ring_i)));
            N = 4 * ring_i;
        } else if (ring_i > 3 * nside) {
            int mirror_i = 4 * nside - ring_i;
            phi0 = expf(LOG_PI - LOG_4 - logf(float(mirror_i)));
            N = 4 * mirror_i;
        } else {
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0 = expf(LOG_PI - LOG_2 - log_nside) * (1.0f - s / 2.0f);
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
    constexpr float LOG_PI = 1.14472988f;  // logf(π)
    float log_N = logf(float(N));

    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        cufftComplex val_Q, val_U;
        if (m <= l_max) {
            size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            float fQ_r = float(fmy_Q_re[idx]);
            float fQ_i = float(fmy_Q_im[idx]);
            float fU_r = float(fmy_U_re[idx]);
            float fU_i = float(fmy_U_im[idx]);

            // phase_angle = m * phi0
            float phase_angle = float(m) * phi0;
            float phase_c, phase_s;
            sincosf(phase_angle, &phase_s, &phase_c);
            float fQ_r_corr = fQ_r * phase_c - fQ_i * phase_s;
            float fQ_i_corr = fQ_r * phase_s + fQ_i * phase_c;
            float fU_r_corr = fU_r * phase_c - fU_i * phase_s;
            float fU_i_corr = fU_r * phase_s + fU_i * phase_c;

            // Use log arithmetic: chirp_angle = π*m²/N = exp(log(π) + 2*log(m) - log(N))
            float chirp_angle = (m == 0) ? 0.0f : expf(LOG_PI + 2.0f * logf(float(m)) - log_N);
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
    constexpr float LOG_PI = 1.14472988f;  // logf(π)
    float log_N = logf(float(N));
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

        // Use log arithmetic: post_angle = π * n² / N = exp(log(π) + 2*log(n) - log(N))
        float post_angle = (n == 0) ? 0.0f : expf(LOG_PI + 2.0f * logf(float(n)) - log_N);
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
