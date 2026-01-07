/**
 * map2alm_v6_log: LOG accumulation mode for map2alm
 *
 * This kernel uses full log-space computation throughout:
 *   - Log-space Ylm recurrence (same as LINEAR mode)
 *   - Log-space Gm coefficients
 *   - Logsumexp accumulation
 *
 * The loop structure is identical to LINEAR mode: O(l_max) per ring batch.
 * The only difference is Gm storage (log vs linear) and accumulation method.
 *
 * Pipeline:
 *   Phase 1: Ring DFT -> Linear Gm -> Convert to Log Gm
 *   Phase 2: Ylm recurrence (log-space) + logsumexp accumulation
 *   Output: Linear alm (converted from log at store)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../include/spht_types.h"
#include "../include/spht_transform_config.cuh"
#include "../include/log_arithmetic.cuh"
#include <stdio.h>
#include <type_traits>

// Forward declaration of DFT kernel for Phase 1 (defined in map2alm_v6.cu)
template<typename T, typename R>
__global__ void compute_gm_kernel_v6(
    int nside, int l_max, int n_maps, int n_rings,
    const T* __restrict__ map_in,
    R* __restrict__ Gm_north_re, R* __restrict__ Gm_north_im,
    R* __restrict__ Gm_south_re, R* __restrict__ Gm_south_im,
    R* __restrict__ cos_theta, R* __restrict__ sin_theta
);

// ============================================================================
// Workspace for LOG mode
// ============================================================================

template<typename T, typename R>
struct Map2almLogWorkspace {
    // Linear Gm buffers (from Phase 1)
    R* Gm_north_re = nullptr;
    R* Gm_north_im = nullptr;
    R* Gm_south_re = nullptr;
    R* Gm_south_im = nullptr;

    // Log-space Gm buffers
    R* Gm_north_log_re = nullptr;
    R* Gm_north_log_im = nullptr;
    R* Gm_south_log_re = nullptr;
    R* Gm_south_log_im = nullptr;
    int8_t* Gm_north_sign_re = nullptr;
    int8_t* Gm_north_sign_im = nullptr;
    int8_t* Gm_south_sign_re = nullptr;
    int8_t* Gm_south_sign_im = nullptr;

    // Geometry
    R* cos_theta = nullptr;
    R* sin_theta = nullptr;

    // Cached sizes
    size_t gm_size = 0;
    size_t geom_size = 0;

    void ensure_size(size_t new_gm, size_t new_geom) {
        if (new_gm > gm_size) {
            // Free old buffers
            if (Gm_north_re) cudaFree(Gm_north_re);
            if (Gm_north_im) cudaFree(Gm_north_im);
            if (Gm_south_re) cudaFree(Gm_south_re);
            if (Gm_south_im) cudaFree(Gm_south_im);
            if (Gm_north_log_re) cudaFree(Gm_north_log_re);
            if (Gm_north_log_im) cudaFree(Gm_north_log_im);
            if (Gm_south_log_re) cudaFree(Gm_south_log_re);
            if (Gm_south_log_im) cudaFree(Gm_south_log_im);
            if (Gm_north_sign_re) cudaFree(Gm_north_sign_re);
            if (Gm_north_sign_im) cudaFree(Gm_north_sign_im);
            if (Gm_south_sign_re) cudaFree(Gm_south_sign_re);
            if (Gm_south_sign_im) cudaFree(Gm_south_sign_im);

            // Allocate linear Gm buffers
            cudaMalloc(&Gm_north_re, new_gm);
            cudaMalloc(&Gm_north_im, new_gm);
            cudaMalloc(&Gm_south_re, new_gm);
            cudaMalloc(&Gm_south_im, new_gm);

            // Allocate log-space Gm buffers
            cudaMalloc(&Gm_north_log_re, new_gm);
            cudaMalloc(&Gm_north_log_im, new_gm);
            cudaMalloc(&Gm_south_log_re, new_gm);
            cudaMalloc(&Gm_south_log_im, new_gm);

            // Sign buffers (int8_t, so size is num_elements)
            size_t sign_size = new_gm / sizeof(R);
            cudaMalloc(&Gm_north_sign_re, sign_size);
            cudaMalloc(&Gm_north_sign_im, sign_size);
            cudaMalloc(&Gm_south_sign_re, sign_size);
            cudaMalloc(&Gm_south_sign_im, sign_size);

            gm_size = new_gm;
        }
        if (new_geom > geom_size) {
            if (cos_theta) cudaFree(cos_theta);
            if (sin_theta) cudaFree(sin_theta);
            cudaMalloc(&cos_theta, new_geom);
            cudaMalloc(&sin_theta, new_geom);
            geom_size = new_geom;
        }
    }

    ~Map2almLogWorkspace() {
        if (Gm_north_re) cudaFree(Gm_north_re);
        if (Gm_north_im) cudaFree(Gm_north_im);
        if (Gm_south_re) cudaFree(Gm_south_re);
        if (Gm_south_im) cudaFree(Gm_south_im);
        if (Gm_north_log_re) cudaFree(Gm_north_log_re);
        if (Gm_north_log_im) cudaFree(Gm_north_log_im);
        if (Gm_south_log_re) cudaFree(Gm_south_log_re);
        if (Gm_south_log_im) cudaFree(Gm_south_log_im);
        if (Gm_north_sign_re) cudaFree(Gm_north_sign_re);
        if (Gm_north_sign_im) cudaFree(Gm_north_sign_im);
        if (Gm_south_sign_re) cudaFree(Gm_south_sign_re);
        if (Gm_south_sign_im) cudaFree(Gm_south_sign_im);
        if (cos_theta) cudaFree(cos_theta);
        if (sin_theta) cudaFree(sin_theta);
    }
};

// Global workspace instances
static Map2almLogWorkspace<double, double> g_m2a_log_ws_f64_f64;
static Map2almLogWorkspace<double, float>  g_m2a_log_ws_f64_f32;
static Map2almLogWorkspace<float, double>  g_m2a_log_ws_f32_f64;
static Map2almLogWorkspace<float, float>   g_m2a_log_ws_f32_f32;

template<typename T, typename R>
Map2almLogWorkspace<T, R>& get_map2alm_log_workspace();

template<> Map2almLogWorkspace<double, double>& get_map2alm_log_workspace<double, double>() { return g_m2a_log_ws_f64_f64; }
template<> Map2almLogWorkspace<double, float>&  get_map2alm_log_workspace<double, float>()  { return g_m2a_log_ws_f64_f32; }
template<> Map2almLogWorkspace<float, double>&  get_map2alm_log_workspace<float, double>()  { return g_m2a_log_ws_f32_f64; }
template<> Map2almLogWorkspace<float, float>&   get_map2alm_log_workspace<float, float>()   { return g_m2a_log_ws_f32_f32; }

// ============================================================================
// Conversion Kernel: Linear Gm -> Log-Cartesian Gm
// ============================================================================

template<typename T, typename R>
__global__ void convert_gm_to_log_kernel(
    int n_elements,
    const T* __restrict__ gm_re_in,
    const T* __restrict__ gm_im_in,
    R* __restrict__ log_re_out,
    R* __restrict__ log_im_out,
    int8_t* __restrict__ sign_re_out,
    int8_t* __restrict__ sign_im_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    R re = R(gm_re_in[idx]);
    R im = R(gm_im_in[idx]);

    log_re_out[idx] = safe_log_typed<R>(re);
    log_im_out[idx] = safe_log_typed<R>(im);
    sign_re_out[idx] = sign_of<R>(re);
    sign_im_out[idx] = sign_of<R>(im);
}

// ============================================================================
// Phase 2 LOG Kernel - O(l_max) structure matching LINEAR mode
// ============================================================================

#define LOG_RING_BATCH_SIZE 64
#define LOG_RINGS_PER_LANE 4

/**
 * Log-space accumulator state for a single alm coefficient
 */
template<typename C>
struct LogAlmAccumulator {
    C log_re;
    C log_im;
    int8_t sign_re;
    int8_t sign_im;
    bool initialized;

    __device__ __forceinline__ void reset() {
        log_re = LogArithmeticTraits<C>::LOG_MIN;
        log_im = LogArithmeticTraits<C>::LOG_MIN;
        sign_re = 0;
        sign_im = 0;
        initialized = false;
    }

    // Accumulate: acc += Ylm * Gm * weight (all in log-space)
    __device__ __forceinline__ void accumulate(
        C log_Ylm, int8_t sign_Ylm,
        C log_Gm_re, int8_t sign_Gm_re,
        C log_Gm_im, int8_t sign_Gm_im,
        C log_weight
    ) {
        // contrib_re = Ylm * Gm_re * weight
        C log_contrib_re = log_Ylm + log_Gm_re + log_weight;
        int8_t sign_contrib_re = sign_Ylm * sign_Gm_re;

        // contrib_im = Ylm * Gm_im * weight
        C log_contrib_im = log_Ylm + log_Gm_im + log_weight;
        int8_t sign_contrib_im = sign_Ylm * sign_Gm_im;

        if (!initialized) {
            log_re = log_contrib_re;
            log_im = log_contrib_im;
            sign_re = sign_contrib_re;
            sign_im = sign_contrib_im;
            initialized = true;
        } else {
            logsumexp_typed<C>(log_re, log_contrib_re, sign_re, sign_contrib_re,
                               &log_re, &sign_re);
            logsumexp_typed<C>(log_im, log_contrib_im, sign_im, sign_contrib_im,
                               &log_im, &sign_im);
        }
    }

    // Merge another accumulator into this one
    __device__ __forceinline__ void merge(const LogAlmAccumulator<C>& other) {
        if (!other.initialized) return;
        if (!initialized) {
            *this = other;
            return;
        }
        logsumexp_typed<C>(log_re, other.log_re, sign_re, other.sign_re,
                           &log_re, &sign_re);
        logsumexp_typed<C>(log_im, other.log_im, sign_im, other.sign_im,
                           &log_im, &sign_im);
    }

    // Convert to linear and store
    template<typename T>
    __device__ __forceinline__ void store(T* out_re, T* out_im) const {
        using Traits = LogArithmeticTraits<C>;
        C re_val = sign_re * Traits::exp_d(clamp_log_typed<C>(log_re));
        C im_val = sign_im * Traits::exp_d(clamp_log_typed<C>(log_im));
        *out_re = T(re_val);
        *out_im = T(im_val);
    }
};

/**
 * Warp-level reduction for LogAlmAccumulator using shuffle
 */
template<typename C>
__device__ __forceinline__ void warp_reduce_log_accum(LogAlmAccumulator<C>* acc) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        LogAlmAccumulator<C> other;
        other.log_re = __shfl_down_sync(0xffffffff, acc->log_re, offset);
        other.log_im = __shfl_down_sync(0xffffffff, acc->log_im, offset);
        other.sign_re = __shfl_down_sync(0xffffffff, acc->sign_re, offset);
        other.sign_im = __shfl_down_sync(0xffffffff, acc->sign_im, offset);
        other.initialized = __shfl_down_sync(0xffffffff, acc->initialized ? 1 : 0, offset);
        acc->merge(other);
    }
}

/**
 * LOG mode Phase 2 kernel with O(l_max) structure (matching LINEAR mode).
 *
 * Structure:
 *   for each batch of rings:
 *       initialize Ymm (once per batch)
 *       for l = m to l_max:           // O(l_max) - single pass
 *           compute Ylm via recurrence
 *           accumulate using logsumexp
 */
template<typename T, typename R>
__global__ void reduce_to_alm_log_kernel_v2(
    int nside, int l_max, int n_maps, int n_north_rings,
    const R* __restrict__ Gm_north_log_re,
    const R* __restrict__ Gm_north_log_im,
    const int8_t* __restrict__ Gm_north_sign_re,
    const int8_t* __restrict__ Gm_north_sign_im,
    const R* __restrict__ Gm_south_log_re,
    const R* __restrict__ Gm_south_log_im,
    const int8_t* __restrict__ Gm_south_sign_re,
    const int8_t* __restrict__ Gm_south_sign_im,
    const R* __restrict__ cos_theta,
    const R* __restrict__ sin_theta,
    R log_pix_area,
    T* __restrict__ alm_out_re,
    T* __restrict__ alm_out_im
) {
    using C = R;
    using Traits = LogArithmeticTraits<C>;

    int m = blockIdx.x;
    int lane = threadIdx.x;
    int lp1 = l_max + 1;

    if (m > l_max || lane >= 32) return;

    // Shared memory for geometry
    extern __shared__ char smem[];
    R* sh_cos_th = (R*)smem;
    R* sh_sin_th = sh_cos_th + LOG_RING_BATCH_SIZE;

    // Per-lane Ylm state (log-space)
    C log_Ylm_prev1[LOG_RINGS_PER_LANE];
    C log_Ylm_prev2[LOG_RINGS_PER_LANE];
    int8_t sign_prev1[LOG_RINGS_PER_LANE];
    int8_t sign_prev2[LOG_RINGS_PER_LANE];
    C log_cos_th_cached[LOG_RINGS_PER_LANE];
    int8_t sign_cos_th_cached[LOG_RINGS_PER_LANE];

    // Per-l accumulators (one per l value)
    // We'll use shared memory for these since l_max can be large
    LogAlmAccumulator<C>* sh_accum = (LogAlmAccumulator<C>*)(sh_sin_th + LOG_RING_BATCH_SIZE);

    // Precompute m-dependent constants
    const C log_prefact = compute_log_prefact_ymm<C>(m);
    const C log_norm = -C(0.5) * Traits::log_d(C(4.0) * Traits::PI_VAL);
    const C log_recur_C_m1 = C(0.5) * Traits::log_d(C(2*m + 3));

    // Process each map separately
    for (int map_idx = 0; map_idx < n_maps; map_idx++) {
        size_t gm_base_idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings;

        // Initialize all l accumulators (cooperatively)
        for (int l = m + lane; l <= l_max; l += 32) {
            sh_accum[l - m].reset();
        }
        __syncwarp();

        // Process ring batches - O(l_max) per batch
        for (int batch_start = 0; batch_start < n_north_rings; batch_start += LOG_RING_BATCH_SIZE) {
            int batch_end = min(batch_start + LOG_RING_BATCH_SIZE, n_north_rings);
            int batch_size = batch_end - batch_start;

            // Load geometry cooperatively
            for (int r = lane; r < batch_size; r += 32) {
                sh_cos_th[r] = cos_theta[batch_start + r];
                sh_sin_th[r] = sin_theta[batch_start + r];
            }
            __syncwarp();

            // Determine this lane's rings
            int rings_per_lane = (batch_size + 31) / 32;
            int my_ring_count = min(rings_per_lane, LOG_RINGS_PER_LANE);

            // Initialize Ymm for rings in this batch (ONCE per batch)
            for (int k = 0; k < my_ring_count; k++) {
                int local_r = lane + 32 * k;
                if (local_r >= batch_size) break;

                C sin_th = C(sh_sin_th[local_r]);
                C cos_th = C(sh_cos_th[local_r]);

                log_cos_th_cached[k] = safe_log_typed<C>(cos_th);
                sign_cos_th_cached[k] = sign_of<C>(cos_th);

                C log_sin_th = safe_log_typed<C>(sin_th);
                C log_Ymm = C(m) * log_sin_th + log_prefact + log_norm;
                int8_t sign_Ymm = ((m & 1) == 0) ? int8_t(1) : int8_t(-1);

                log_Ylm_prev1[k] = log_Ymm;
                sign_prev1[k] = sign_Ymm;
                log_Ylm_prev2[k] = Traits::LOG_MIN;
                sign_prev2[k] = 0;
            }

            // SINGLE PASS over all l values - O(l_max) complexity
            for (int l = m; l <= l_max; l++) {
                // Per-lane accumulator for this l
                LogAlmAccumulator<C> lane_acc;
                lane_acc.reset();

                // Process all rings for this lane
                for (int k = 0; k < my_ring_count; k++) {
                    int local_r = lane + 32 * k;
                    if (local_r >= batch_size) break;
                    int global_r = batch_start + local_r;

                    C log_Ylm;
                    int8_t sign_Ylm;

                    if (l == m) {
                        log_Ylm = log_Ylm_prev1[k];
                        sign_Ylm = sign_prev1[k];
                    } else if (l == m + 1) {
                        log_Ylm = log_cos_th_cached[k] + log_recur_C_m1 + log_Ylm_prev1[k];
                        sign_Ylm = sign_cos_th_cached[k] * sign_prev1[k];

                        log_Ylm_prev2[k] = log_Ylm_prev1[k];
                        sign_prev2[k] = sign_prev1[k];
                        log_Ylm_prev1[k] = log_Ylm;
                        sign_prev1[k] = sign_Ylm;
                    } else {
                        C log_A = log_A_lm<C>(l, m);
                        C log_B = log_B_lm<C>(l, m);
                        C R1 = log_A + log_cos_th_cached[k] + log_Ylm_prev1[k];
                        int8_t S1 = sign_cos_th_cached[k] * sign_prev1[k];
                        C R2 = log_B + log_Ylm_prev2[k];
                        int8_t S2 = -sign_prev2[k];

                        logsumexp_fast<C>(R1, R2, S1, S2, &log_Ylm, &sign_Ylm);

                        log_Ylm_prev2[k] = log_Ylm_prev1[k];
                        sign_prev2[k] = sign_prev1[k];
                        log_Ylm_prev1[k] = log_Ylm;
                        sign_prev1[k] = sign_Ylm;
                    }

                    // Load log-space Gm values
                    size_t idx = gm_base_idx + global_r;

                    // North contribution
                    lane_acc.accumulate(log_Ylm, sign_Ylm,
                                        Gm_north_log_re[idx], Gm_north_sign_re[idx],
                                        Gm_north_log_im[idx], Gm_north_sign_im[idx],
                                        log_pix_area);

                    // South contribution: Y_l^m(pi-theta) = (-1)^(l+m) * Y_l^m(theta)
                    int8_t south_sign = ((l + m) & 1) ? -sign_Ylm : sign_Ylm;
                    lane_acc.accumulate(log_Ylm, south_sign,
                                        Gm_south_log_re[idx], Gm_south_sign_re[idx],
                                        Gm_south_log_im[idx], Gm_south_sign_im[idx],
                                        log_pix_area);
                }

                // Warp reduce for this l
                warp_reduce_log_accum(&lane_acc);

                // Lane 0 merges into shared accumulator
                if (lane == 0) {
                    sh_accum[l - m].merge(lane_acc);
                }
                __syncwarp();
            }
        }

        // Write results
        if (lane == 0) {
            T* alm_re_t = alm_out_re + (size_t)map_idx * lp1 * lp1;
            T* alm_im_t = alm_out_im + (size_t)map_idx * lp1 * lp1;
            for (int l = m; l <= l_max; l++) {
                sh_accum[l - m].store(&alm_re_t[l * lp1 + m], &alm_im_t[l * lp1 + m]);
            }
        }
    }
}

// ============================================================================
// Main Implementation Function
// ============================================================================

template<typename T, typename R>
void map2alm_cuda_v6_log_impl(
    int nside, int l_max, int n_maps,
    const T* map_in,
    T* alm_out_re, T* alm_out_im
) {
    int n_rings = 4 * nside - 1;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;

    // Get workspace
    auto& ws = get_map2alm_log_workspace<T, R>();

    // Compute buffer sizes
    size_t gm_elements = (size_t)n_maps * lp1 * n_north_rings;
    size_t gm_bytes = gm_elements * sizeof(R);
    size_t geom_bytes = n_north_rings * sizeof(R);

    // Ensure workspace is large enough
    ws.ensure_size(gm_bytes, geom_bytes);

    // Timing
    static bool timing_enabled = (getenv("SPHT_TIMING") != nullptr);
    cudaEvent_t start_p1, end_p1, start_conv, end_conv, start_p2, end_p2;
    if (timing_enabled) {
        cudaEventCreate(&start_p1);
        cudaEventCreate(&end_p1);
        cudaEventCreate(&start_conv);
        cudaEventCreate(&end_conv);
        cudaEventCreate(&start_p2);
        cudaEventCreate(&end_p2);
        cudaEventRecord(start_p1);
    }

    // ================================================================
    // Phase 1: Compute Gm using DFT (reuse from LINEAR mode)
    // ================================================================

    int block_size = min(256, lp1);
    compute_gm_kernel_v6<T, R><<<n_north_rings, block_size>>>(
        nside, l_max, n_maps, n_rings, map_in,
        ws.Gm_north_re, ws.Gm_north_im,
        ws.Gm_south_re, ws.Gm_south_im,
        ws.cos_theta, ws.sin_theta
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "map2alm_log Phase 1 error: %s\n", cudaGetErrorString(err));
        return;
    }

    if (timing_enabled) {
        cudaEventRecord(end_p1);
        cudaEventRecord(start_conv);
    }

    // ================================================================
    // Convert Gm: Linear -> Log-Cartesian
    // ================================================================

    int conv_block = 256;
    int conv_grid = (gm_elements + conv_block - 1) / conv_block;

    // Convert north Gm
    convert_gm_to_log_kernel<R, R><<<conv_grid, conv_block>>>(
        gm_elements,
        ws.Gm_north_re, ws.Gm_north_im,
        ws.Gm_north_log_re, ws.Gm_north_log_im,
        ws.Gm_north_sign_re, ws.Gm_north_sign_im
    );

    // Convert south Gm
    convert_gm_to_log_kernel<R, R><<<conv_grid, conv_block>>>(
        gm_elements,
        ws.Gm_south_re, ws.Gm_south_im,
        ws.Gm_south_log_re, ws.Gm_south_log_im,
        ws.Gm_south_sign_re, ws.Gm_south_sign_im
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "map2alm_log conversion error: %s\n", cudaGetErrorString(err));
        return;
    }

    if (timing_enabled) {
        cudaEventRecord(end_conv);
        cudaEventRecord(start_p2);
    }

    // ================================================================
    // Phase 2: LOG accumulation kernel
    // ================================================================

    R pix_area = R(4.0 * M_PI / (12.0 * nside * nside));
    R log_pix_area = log(pix_area);

    // Shared memory: geometry + accumulators
    size_t smem_size = 2 * LOG_RING_BATCH_SIZE * sizeof(R) +
                       (l_max + 1) * sizeof(LogAlmAccumulator<R>);

    reduce_to_alm_log_kernel_v2<T, R><<<lp1, 32, smem_size>>>(
        nside, l_max, n_maps, n_north_rings,
        ws.Gm_north_log_re, ws.Gm_north_log_im,
        ws.Gm_north_sign_re, ws.Gm_north_sign_im,
        ws.Gm_south_log_re, ws.Gm_south_log_im,
        ws.Gm_south_sign_re, ws.Gm_south_sign_im,
        ws.cos_theta, ws.sin_theta,
        log_pix_area,
        alm_out_re, alm_out_im
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "map2alm_log Phase 2 error: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaDeviceSynchronize();

    if (timing_enabled) {
        cudaEventRecord(end_p2);
        cudaEventSynchronize(end_p2);

        float p1_ms, conv_ms, p2_ms;
        cudaEventElapsedTime(&p1_ms, start_p1, end_p1);
        cudaEventElapsedTime(&conv_ms, start_conv, end_conv);
        cudaEventElapsedTime(&p2_ms, start_p2, end_p2);

        fprintf(stderr, "[SPHT_TIMING] LOG mode: nside=%d l_max=%d n_maps=%d P1=%.2fms Conv=%.2fms P2=%.2fms Total=%.2fms\n",
                nside, l_max, n_maps, p1_ms, conv_ms, p2_ms, p1_ms + conv_ms + p2_ms);

        cudaEventDestroy(start_p1);
        cudaEventDestroy(end_p1);
        cudaEventDestroy(start_conv);
        cudaEventDestroy(end_conv);
        cudaEventDestroy(start_p2);
        cudaEventDestroy(end_p2);
    }
}

// ============================================================================
// C API Entry Points
// ============================================================================

extern "C" {

void map2alm_cuda_v6_log_f64_f64(
    int nside, int l_max, int n_maps,
    const double* map_in,
    double* alm_re_out, double* alm_im_out
) {
    map2alm_cuda_v6_log_impl<double, double>(nside, l_max, n_maps, map_in, alm_re_out, alm_im_out);
}

void map2alm_cuda_v6_log_f64_f32(
    int nside, int l_max, int n_maps,
    const double* map_in,
    double* alm_re_out, double* alm_im_out
) {
    map2alm_cuda_v6_log_impl<double, float>(nside, l_max, n_maps, map_in, alm_re_out, alm_im_out);
}

void map2alm_cuda_v6_log_f32_f64(
    int nside, int l_max, int n_maps,
    const float* map_in,
    float* alm_re_out, float* alm_im_out
) {
    map2alm_cuda_v6_log_impl<float, double>(nside, l_max, n_maps, map_in, alm_re_out, alm_im_out);
}

void map2alm_cuda_v6_log_f32_f32(
    int nside, int l_max, int n_maps,
    const float* map_in,
    float* alm_re_out, float* alm_im_out
) {
    map2alm_cuda_v6_log_impl<float, float>(nside, l_max, n_maps, map_in, alm_re_out, alm_im_out);
}

}  // extern "C"
