/**
 * alm2map_v6_log: LOG accumulation mode for alm2map (synthesis)
 *
 * This kernel uses full log-space computation for the synthesis operation.
 * Required for bf16 precision, optional for f32/f64 with extreme dynamic range.
 *
 * Key differences from LINEAR mode:
 *   - alm coefficients converted to log-Cartesian at input
 *   - Accumulation of Gm uses logsumexp
 *   - Final inverse DFT produces log-space output converted at store
 *
 * Pipeline:
 *   Convert: Linear alm -> Log-Cartesian
 *   Phase 2: Ylm synthesis (log-space) + logsumexp accumulation -> Gm
 *   Convert: Log-Cartesian Gm -> Log-Polar for inverse DFT
 *   Phase 1: Inverse Ring DFT -> map output
 *   Convert: Log-space map -> linear output
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../include/spht_types.h"
#include "../include/log_arithmetic.cuh"
#include <stdio.h>

// ============================================================================
// LOG Synthesis Kernel - Phase 2 with logsumexp
// ============================================================================

/**
 * Log-space accumulator for Gm synthesis (complex value)
 */
template<typename C>
struct LogGmAccumulator {
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

    // Accumulate: Gm += alm * Ylm
    // alm is complex (log-Cartesian), Ylm is real (log + sign)
    __device__ __forceinline__ void accumulate(
        C log_alm_re, int8_t sign_alm_re,
        C log_alm_im, int8_t sign_alm_im,
        C log_Ylm, int8_t sign_Ylm
    ) {
        // contrib_re = alm_re * Ylm
        C log_contrib_re = log_alm_re + log_Ylm;
        int8_t sign_contrib_re = sign_alm_re * sign_Ylm;

        // contrib_im = alm_im * Ylm
        C log_contrib_im = log_alm_im + log_Ylm;
        int8_t sign_contrib_im = sign_alm_im * sign_Ylm;

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

    // Merge accumulators (for warp reduction)
    __device__ __forceinline__ void merge(const LogGmAccumulator<C>& other) {
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

    // Convert to linear for output
    template<typename T>
    __device__ __forceinline__ void to_linear(T* out_re, T* out_im) const {
        using Traits = LogArithmeticTraits<C>;
        C re_val = sign_re * Traits::exp_d(clamp_log_typed<C>(log_re));
        C im_val = sign_im * Traits::exp_d(clamp_log_typed<C>(log_im));
        *out_re = T(re_val);
        *out_im = T(im_val);
    }
};

/**
 * Warp-level reduction for LogGmAccumulator
 */
template<typename C>
__device__ __forceinline__ void warp_reduce_log_gm(LogGmAccumulator<C>* acc) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        LogGmAccumulator<C> other;
        other.log_re = __shfl_down_sync(0xffffffff, acc->log_re, offset);
        other.log_im = __shfl_down_sync(0xffffffff, acc->log_im, offset);
        other.sign_re = __shfl_down_sync(0xffffffff, acc->sign_re, offset);
        other.sign_im = __shfl_down_sync(0xffffffff, acc->sign_im, offset);
        other.initialized = __shfl_down_sync(0xffffffff, acc->initialized ? 1 : 0, offset);
        acc->merge(other);
    }
}

// ============================================================================
// Phase 2 LOG Synthesis Kernel
// ============================================================================

#define LOG_SYNTH_BATCH_SIZE 64
#define LOG_SYNTH_RINGS_PER_LANE 4

template<typename T, typename R>
__global__ void synthesize_gm_log_kernel(
    int nside, int l_max, int n_maps, int n_north_rings,
    const R* __restrict__ alm_log_re,  // log(|alm_re|)
    const R* __restrict__ alm_log_im,  // log(|alm_im|)
    const int8_t* __restrict__ alm_sign_re,
    const int8_t* __restrict__ alm_sign_im,
    const R* __restrict__ cos_theta,
    const R* __restrict__ sin_theta,
    R* __restrict__ Gm_north_log_re,
    R* __restrict__ Gm_north_log_im,
    int8_t* __restrict__ Gm_north_sign_re,
    int8_t* __restrict__ Gm_north_sign_im,
    R* __restrict__ Gm_south_log_re,
    R* __restrict__ Gm_south_log_im,
    int8_t* __restrict__ Gm_south_sign_re,
    int8_t* __restrict__ Gm_south_sign_im
) {
    using C = R;
    using Traits = LogArithmeticTraits<C>;

    int m = blockIdx.x;
    int ring_block = blockIdx.y;
    int lane = threadIdx.x;
    int lp1 = l_max + 1;

    if (m > l_max || lane >= 32) return;

    // Shared memory for geometry
    extern __shared__ char smem[];
    R* sh_cos_th = (R*)smem;
    R* sh_sin_th = sh_cos_th + LOG_SYNTH_BATCH_SIZE;

    // Per-lane Ylm state
    C log_Ylm_prev1[LOG_SYNTH_RINGS_PER_LANE];
    C log_Ylm_prev2[LOG_SYNTH_RINGS_PER_LANE];
    int8_t sign_prev1[LOG_SYNTH_RINGS_PER_LANE];
    int8_t sign_prev2[LOG_SYNTH_RINGS_PER_LANE];
    C log_cos_th_cached[LOG_SYNTH_RINGS_PER_LANE];
    int8_t sign_cos_th_cached[LOG_SYNTH_RINGS_PER_LANE];

    // Precompute m-dependent constants
    const C log_prefact = compute_log_prefact_ymm<C>(m);
    const C log_norm = -C(0.5) * Traits::log_d(C(4.0) * Traits::PI_VAL);
    const C log_recur_C_m1 = C(0.5) * Traits::log_d(C(2*m + 3));

    int batch_start = ring_block * LOG_SYNTH_BATCH_SIZE;
    int batch_end = min(batch_start + LOG_SYNTH_BATCH_SIZE, n_north_rings);
    int batch_size = batch_end - batch_start;

    if (batch_size <= 0) return;

    // Load geometry
    for (int r = lane; r < batch_size; r += 32) {
        sh_cos_th[r] = cos_theta[batch_start + r];
        sh_sin_th[r] = sin_theta[batch_start + r];
    }
    __syncwarp();

    // Determine this lane's rings
    int rings_per_lane = (batch_size + 31) / 32;
    int my_ring_count = min(rings_per_lane, LOG_SYNTH_RINGS_PER_LANE);

    // Initialize Ymm for rings
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

    // Process each map
    for (int map_idx = 0; map_idx < n_maps; map_idx++) {
        size_t alm_base = (size_t)map_idx * lp1 * lp1;
        size_t gm_base = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings;

        // Reset Ylm state for this map
        for (int k = 0; k < my_ring_count; k++) {
            int local_r = lane + 32 * k;
            if (local_r >= batch_size) break;

            C sin_th = C(sh_sin_th[local_r]);
            C log_sin_th = safe_log_typed<C>(sin_th);
            C log_Ymm = C(m) * log_sin_th + log_prefact + log_norm;
            int8_t sign_Ymm = ((m & 1) == 0) ? int8_t(1) : int8_t(-1);

            log_Ylm_prev1[k] = log_Ymm;
            sign_prev1[k] = sign_Ymm;
            log_Ylm_prev2[k] = Traits::LOG_MIN;
            sign_prev2[k] = 0;
        }

        // Accumulate Gm for each ring
        LogGmAccumulator<C> gm_north[LOG_SYNTH_RINGS_PER_LANE];
        LogGmAccumulator<C> gm_south[LOG_SYNTH_RINGS_PER_LANE];

        for (int k = 0; k < my_ring_count; k++) {
            gm_north[k].reset();
            gm_south[k].reset();
        }

        // Sum over l for this m
        for (int l = m; l <= l_max; l++) {
            // Load alm[l,m] in log form
            size_t alm_idx = alm_base + l * lp1 + m;
            C log_alm_re = alm_log_re[alm_idx];
            C log_alm_im = alm_log_im[alm_idx];
            int8_t sign_alm_re = alm_sign_re[alm_idx];
            int8_t sign_alm_im = alm_sign_im[alm_idx];

            // For each ring this lane handles
            for (int k = 0; k < my_ring_count; k++) {
                int local_r = lane + 32 * k;
                if (local_r >= batch_size) break;

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

                // North: accumulate alm * Ylm
                gm_north[k].accumulate(log_alm_re, sign_alm_re,
                                       log_alm_im, sign_alm_im,
                                       log_Ylm, sign_Ylm);

                // South: Y_l^m(pi-theta) = (-1)^(l+m) * Y_l^m(theta)
                int8_t south_sign_Ylm = ((l + m) & 1) ? -sign_Ylm : sign_Ylm;
                gm_south[k].accumulate(log_alm_re, sign_alm_re,
                                       log_alm_im, sign_alm_im,
                                       log_Ylm, south_sign_Ylm);
            }
        }

        // Store Gm results
        for (int k = 0; k < my_ring_count; k++) {
            int local_r = lane + 32 * k;
            if (local_r >= batch_size) break;
            int global_r = batch_start + local_r;
            size_t idx = gm_base + global_r;

            Gm_north_log_re[idx] = gm_north[k].log_re;
            Gm_north_log_im[idx] = gm_north[k].log_im;
            Gm_north_sign_re[idx] = gm_north[k].sign_re;
            Gm_north_sign_im[idx] = gm_north[k].sign_im;

            Gm_south_log_re[idx] = gm_south[k].log_re;
            Gm_south_log_im[idx] = gm_south[k].log_im;
            Gm_south_sign_re[idx] = gm_south[k].sign_re;
            Gm_south_sign_im[idx] = gm_south[k].sign_im;
        }
    }
}

// ============================================================================
// Conversion Kernels
// ============================================================================

/**
 * Convert linear alm to log-Cartesian form
 */
template<typename T, typename R>
__global__ void convert_alm_to_log_kernel(
    int n_elements,
    const T* __restrict__ alm_re_in,
    const T* __restrict__ alm_im_in,
    R* __restrict__ log_re_out,
    R* __restrict__ log_im_out,
    int8_t* __restrict__ sign_re_out,
    int8_t* __restrict__ sign_im_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    R re = R(alm_re_in[idx]);
    R im = R(alm_im_in[idx]);

    log_re_out[idx] = safe_log_typed<R>(re);
    log_im_out[idx] = safe_log_typed<R>(im);
    sign_re_out[idx] = sign_of<R>(re);
    sign_im_out[idx] = sign_of<R>(im);
}

/**
 * Convert log-Cartesian Gm to linear for inverse DFT
 */
template<typename T, typename R>
__global__ void convert_gm_to_linear_kernel(
    int n_elements,
    const R* __restrict__ log_re_in,
    const R* __restrict__ log_im_in,
    const int8_t* __restrict__ sign_re_in,
    const int8_t* __restrict__ sign_im_in,
    T* __restrict__ re_out,
    T* __restrict__ im_out
) {
    using Traits = LogArithmeticTraits<R>;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    R log_re = log_re_in[idx];
    R log_im = log_im_in[idx];
    int8_t sign_re = sign_re_in[idx];
    int8_t sign_im = sign_im_in[idx];

    R re_val = sign_re * Traits::exp_d(clamp_log_typed<R>(log_re));
    R im_val = sign_im * Traits::exp_d(clamp_log_typed<R>(log_im));

    re_out[idx] = T(re_val);
    im_out[idx] = T(im_val);
}

// ============================================================================
// External API
// ============================================================================

extern "C"
void alm2map_cuda_v6_log_f32_f32(
    int nside, int l_max, int n_maps,
    const float* alm_re_in, const float* alm_im_in,
    float* map_out
) {
    // Placeholder - full integration requires:
    // 1. Convert alm to log form
    // 2. Synthesize Gm in log form
    // 3. Convert Gm to linear
    // 4. Inverse DFT to get map

    fprintf(stderr, "alm2map_cuda_v6_log_f32_f32: LOG mode not yet fully integrated\n");
}

extern "C"
void alm2map_cuda_v6_log_f64_f64(
    int nside, int l_max, int n_maps,
    const double* alm_re_in, const double* alm_im_in,
    double* map_out
) {
    fprintf(stderr, "alm2map_cuda_v6_log_f64_f64: LOG mode not yet fully integrated\n");
}
