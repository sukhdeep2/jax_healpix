#ifndef LOGSPACE_MATH_CUH
#define LOGSPACE_MATH_CUH

/**
 * Log-Space Math Primitives
 *
 * Core log-space arithmetic for numerical stability in spherical harmonic transforms.
 * Reference: jax_healpix/utils.py and YLM_jax_log.py
 *
 * Key operations:
 * - logsumexp: log(sign_A * exp(log_A) + sign_B * exp(log_B))
 * - logdiffexp: log(exp(log_a) - exp(log_b)) assuming a > b
 * - safe_log: log(|x|) with zero handling
 * - clamp_log: prevent overflow on exp()
 */

#include <math.h>
#include <stdint.h>

// bf16 support requires sm_80+ (Ampere and later)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define SPHT_HAS_BF16 1
#include <cuda_bf16.h>
#else
#define SPHT_HAS_BF16 0
#endif

// ============================================================================
// Type Traits
// ============================================================================

/**
 * LogArithmeticTraits: Type-specific constants and operations
 *
 * Provides precision-appropriate constants and math functions:
 * - LOG_MIN/MAX: Safe bounds for log values before exp()
 * - log_d/exp_d/log1p_d: Precision-appropriate math intrinsics
 */
template<typename T>
struct LogArithmeticTraits;

template<>
struct LogArithmeticTraits<double> {
    static constexpr double LOG_MIN = -700.0;
    static constexpr double LOG_MAX = 700.0;
    static constexpr double LOG_4PI_VAL = 2.5310242469692907;
    static constexpr double PI_VAL = 3.14159265358979323846;
    static constexpr double LOG2_VAL = 0.6931471805599453;

    static __device__ __forceinline__ double log_d(double x) { return log(x); }
    static __device__ __forceinline__ double exp_d(double x) { return exp(x); }
    static __device__ __forceinline__ double log1p_d(double x) { return log1p(x); }
    static __device__ __forceinline__ double abs_d(double x) { return fabs(x); }
    static __device__ __forceinline__ double sqrt_d(double x) { return sqrt(x); }
    static __device__ __forceinline__ double cos_d(double x) { return cos(x); }
    static __device__ __forceinline__ double sin_d(double x) { return sin(x); }
    static __device__ __forceinline__ double lgamma_d(double x) { return lgamma(x); }
};

template<>
struct LogArithmeticTraits<float> {
    static constexpr float LOG_MIN = -87.0f;
    static constexpr float LOG_MAX = 88.0f;
    static constexpr float LOG_4PI_VAL = 2.53102425f;
    static constexpr float PI_VAL = 3.14159265f;
    static constexpr float LOG2_VAL = 0.6931472f;

    // Use fast intrinsics for maximum performance
    static __device__ __forceinline__ float log_d(float x) { return __logf(x); }
    static __device__ __forceinline__ float exp_d(float x) { return __expf(x); }
    static __device__ __forceinline__ float log1p_d(float x) { return log1pf(x); }
    static __device__ __forceinline__ float abs_d(float x) { return fabsf(x); }
    static __device__ __forceinline__ float sqrt_d(float x) { return __fsqrt_rn(x); }
    static __device__ __forceinline__ float cos_d(float x) { return __cosf(x); }
    static __device__ __forceinline__ float sin_d(float x) { return __sinf(x); }
    static __device__ __forceinline__ float lgamma_d(float x) { return lgammaf(x); }
};

#if SPHT_HAS_BF16
/**
 * bf16 (bfloat16) Traits
 *
 * Key properties:
 * - 8-bit exponent (same range as f32): ~1e-38 to ~3e38
 * - 7-bit mantissa (~2-3 decimal digits precision)
 * - LOG_MIN/MAX same as f32 (same exponent range)
 *
 * Strategy: Store in bf16, compute in f32
 * - Load bf16 -> convert to f32
 * - All math operations in f32
 * - Convert back to bf16 at store
 *
 * This gives 2x memory savings with f32-level compute precision.
 */
template<>
struct LogArithmeticTraits<__nv_bfloat16> {
    // Same exponent range as f32
    static constexpr float LOG_MIN = -87.0f;
    static constexpr float LOG_MAX = 88.0f;
    static constexpr float LOG_4PI_VAL = 2.53102425f;
    static constexpr float PI_VAL = 3.14159265f;
    static constexpr float LOG2_VAL = 0.6931472f;

    // All operations promote to f32 for compute, return f32
    // Conversion to/from bf16 happens at load/store boundaries
    static __device__ __forceinline__ float log_d(__nv_bfloat16 x) {
        return __logf(__bfloat162float(x));
    }
    static __device__ __forceinline__ float exp_d(float x) { return __expf(x); }
    static __device__ __forceinline__ float log1p_d(float x) { return log1pf(x); }
    static __device__ __forceinline__ float abs_d(__nv_bfloat16 x) {
        return fabsf(__bfloat162float(x));
    }
    static __device__ __forceinline__ float sqrt_d(__nv_bfloat16 x) {
        return __fsqrt_rn(__bfloat162float(x));
    }
    static __device__ __forceinline__ float cos_d(float x) { return __cosf(x); }
    static __device__ __forceinline__ float sin_d(float x) { return __sinf(x); }
    static __device__ __forceinline__ float lgamma_d(float x) { return lgammaf(x); }

    // Conversion helpers
    static __device__ __forceinline__ float to_float(__nv_bfloat16 x) {
        return __bfloat162float(x);
    }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float x) {
        return __float2bfloat16(x);
    }
};
#endif // SPHT_HAS_BF16

// ============================================================================
// Core Log-Space Operations
// ============================================================================

/**
 * logdiffexp: Computes log(a - b) given log(a) and log(b), assuming a > b
 *
 * log(a-b) = log(a * (1 - b/a)) = log(a) + log(1 - exp(log_b - log_a))
 */
template<typename T>
__device__ __forceinline__
T logdiffexp_typed(T log_a, T log_b) {
    using Traits = LogArithmeticTraits<T>;
    T diff = log_b - log_a;
    return log_a + Traits::log1p_d(-Traits::exp_d(diff));
}

/**
 * logsumexp: Computes log(sign_A * A + sign_B * B) and resulting sign
 *
 * Given: A = sign_A * exp(log_A), B = sign_B * exp(log_B)
 * Computes: C = A + B = sign_result * exp(log_result)
 *
 * OPTIMIZED: Branchless implementation to avoid warp divergence.
 */
template<typename T>
__device__ __forceinline__
void logsumexp_typed(T log_A, T log_B,
                     int8_t sign_A, int8_t sign_B,
                     T* log_result, int8_t* sign_result) {
    using Traits = LogArithmeticTraits<T>;

    // Compute relative sign for the exponential term
    T relative_sign = T(sign_A * sign_B);

    // Branchless max/min selection
    bool a_larger = log_A >= log_B;
    T max_log = a_larger ? log_A : log_B;
    T min_log = a_larger ? log_B : log_A;

    // Core computation using log1p for stability
    T delta = min_log - max_log;

    // Always compute log1p term, use branchless select for final result
    T log1p_term = Traits::log1p_d(relative_sign * Traits::exp_d(delta));

    // Branchless: if delta < LOG_MIN, log1p_term contribution is ~0
    bool use_log1p = delta >= Traits::LOG_MIN;
    *log_result = max_log + (use_log1p ? log1p_term : T(0));

    // Branchless sign determination:
    // If same signs -> that sign, else larger magnitude's sign
    bool same_sign = (sign_A == sign_B);
    int8_t max_sign = a_larger ? sign_A : sign_B;
    *sign_result = same_sign ? sign_A : max_sign;
}

/**
 * logsumexp_fast: Truly branchless logsumexp for Ylm recurrence
 *
 * Uses multiplication-based selection to avoid ALL branches,
 * including ternary operators that may compile to branches.
 */
template<typename T>
__device__ __forceinline__
void logsumexp_fast(T R1, T R2, int8_t S1, int8_t S2,
                    T* log_result, int8_t* sign_result) {
    using Traits = LogArithmeticTraits<T>;

    // Truly branchless max/min using multiplication
    T r1_mask = T(R1 >= R2);
    T r2_mask = T(1) - r1_mask;

    T max_log = r1_mask * R1 + r2_mask * R2;
    T min_log = r1_mask * R2 + r2_mask * R1;

    T delta = min_log - max_log;
    T rel_sign = T(S1 * S2);

    // Always compute the log1p term
    T exp_delta = Traits::exp_d(delta);
    T log1p_term = Traits::log1p_d(rel_sign * exp_delta);

    // Branchless select using multiplication
    T use_mask = T(delta >= Traits::LOG_MIN);
    *log_result = max_log + use_mask * log1p_term;

    // Sign: branchless using multiplication
    int8_t max_sign = int8_t(r1_mask) * S1 + int8_t(r2_mask) * S2;
    int8_t same_sign_mask = int8_t(S1 == S2);
    *sign_result = same_sign_mask * S1 + (1 - same_sign_mask) * max_sign;
}

/**
 * safe_log: Returns log(|x|), handling x == 0 gracefully
 * Returns LOG_MIN for x == 0
 */
template<typename T>
__device__ __forceinline__
T safe_log_typed(T x) {
    using Traits = LogArithmeticTraits<T>;
    T abs_x = Traits::abs_d(x);
    if (abs_x <= T(0)) {
        return Traits::LOG_MIN;
    }
    return Traits::log_d(abs_x);
}

/**
 * sign_of: Returns +1 for x >= 0, -1 for x < 0
 */
template<typename T>
__device__ __forceinline__
int8_t sign_of(T x) {
    return (x >= T(0)) ? int8_t(1) : int8_t(-1);
}

/**
 * clamp_log: Clamp log values to prevent overflow on exp()
 */
template<typename T>
__device__ __forceinline__
T clamp_log_typed(T log_val) {
    using Traits = LogArithmeticTraits<T>;
    if (log_val < Traits::LOG_MIN) return Traits::LOG_MIN;
    if (log_val > Traits::LOG_MAX) return Traits::LOG_MAX;
    return log_val;
}

// ============================================================================
// Ylm Recurrence Coefficients
// ============================================================================

/**
 * log_A_lm: Compute log of recurrence coefficient A_{l,m}
 *
 * Reference: jax_healpix/YLM_jax_log.py lines 66-73
 * Eq. 14 of arXiv:1010.2084
 *
 * A_{l,m} = sqrt((4l^2 - 1) / (l^2 - m^2))
 * log(A) = 0.5 * [log(4l^2 - 1) - log(l^2 - m^2)]
 */
template<typename T>
__device__ __forceinline__
T log_A_lm(int l, int m) {
    using Traits = LogArithmeticTraits<T>;
    T l_c = T(l);
    T m_c = T(m);
    T l2 = l_c * l_c;
    T m2 = m_c * m_c;

    T log_num = Traits::log_d(T(4) * l2 - T(1));
    T log_denom = Traits::log_d(l2 - m2);

    return T(0.5) * (log_num - log_denom);
}

/**
 * log_B_lm: Compute log of recurrence coefficient B_{l,m} = A_{l,m} / A_{l-1,m}
 *
 * B_{l,m} = sqrt((2l+1)/(2l-3) * ((l-1)^2 - m^2) / (l^2 - m^2))
 */
template<typename T>
__device__ __forceinline__
T log_B_lm(int l, int m) {
    using Traits = LogArithmeticTraits<T>;
    T l_c = T(l);
    T m_c = T(m);
    T l2 = l_c * l_c;
    T m2 = m_c * m_c;
    T lm1_2 = T(l - 1) * T(l - 1);

    T num = (T(2) * l_c + T(1)) * (lm1_2 - m2);
    T denom = (T(2) * l_c - T(3)) * (l2 - m2);

    return T(0.5) * Traits::log_d(num / denom);
}

/**
 * ABRecurrenceState: Incremental A/B coefficient computation
 *
 * Instead of recomputing log_A_lm and log_B_lm from scratch each iteration,
 * tracks log_A[l-1] and updates incrementally. Saves ~8 ops per (l,m) iteration.
 */
template<typename T>
struct ABRecurrenceState {
    T log_A_prev;  // log(A[l-1,m])
    int m;         // Current m value (fixed for the recurrence)

    // Initialize at l=m+1 (first valid l for 3-term recurrence)
    __device__ __forceinline__
    void init(int m_val) {
        m = m_val;
        log_A_prev = log_A_lm<T>(m + 1, m);
    }

    // Compute log_A[l] and log_B[l] incrementally, advance state
    // Returns log_B[l] = log_A[l] - log_A[l-1]
    __device__ __forceinline__
    T step(int l, T* log_A_out) {
        using Traits = LogArithmeticTraits<T>;
        T l_f = T(l);
        T m_f = T(m);

        // Compute ratio: A[l]² / A[l-1]² in log-space
        T l4_term = T(4) * l_f * l_f - T(1);
        T lm1_4_term = T(4) * (l_f - T(1)) * (l_f - T(1)) - T(1);

        T log_ratio = T(0.5) * (
            Traits::log_d(l4_term) - Traits::log_d(lm1_4_term)
            + Traits::log_d(l_f - T(1) - m_f) + Traits::log_d(l_f - T(1) + m_f)
            - Traits::log_d(l_f - m_f) - Traits::log_d(l_f + m_f)
        );

        T log_A = log_A_prev + log_ratio;
        T log_B = log_ratio;  // B[l] = A[l] / A[l-1]

        // Advance state
        log_A_prev = log_A;

        *log_A_out = log_A;
        return log_B;
    }
};

/**
 * compute_log_prefact_ymm: Cumulative log prefactor for Y[m,m] initialization
 *
 * prefact[m] = prod_{j=1}^{m} sqrt((2j+1)/(2j))
 *
 * Using lgamma for O(1) computation instead of O(m) loop:
 * log_prefact = 0.5 * [lgamma(2m+2) - 2m*log(2) - 2*lgamma(m+1)]
 */
template<typename T>
__device__ __forceinline__
T compute_log_prefact_ymm(int m) {
    using Traits = LogArithmeticTraits<T>;
    if (m == 0) return T(0);

    T m_c = T(m);
    T lgamma_2m2 = Traits::lgamma_d(T(2) * m_c + T(2));
    T lgamma_m1 = Traits::lgamma_d(m_c + T(1));

    return T(0.5) * (lgamma_2m2 - T(2) * m_c * Traits::LOG2_VAL - T(2) * lgamma_m1);
}

#endif // LOGSPACE_MATH_CUH
