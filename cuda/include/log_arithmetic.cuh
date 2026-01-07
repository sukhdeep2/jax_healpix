#ifndef LOG_ARITHMETIC_CUH
#define LOG_ARITHMETIC_CUH

#include "spht_types.h"
#include <math.h>

/**
 * Log-Arithmetic Traits for type-specific operations
 * Reference: jax_healpix/utils.py and YLM_jax_log.py
 */
template<typename C>
struct LogArithmeticTraits;

template<>
struct LogArithmeticTraits<double> {
    static constexpr double LOG_MIN = -700.0;
    static constexpr double LOG_MAX = 700.0;
    static constexpr double LOG_4PI_VAL = 2.5310242469692907;
    static constexpr double PI_VAL = 3.14159265358979323846;

    static __device__ __forceinline__ double log_d(double x) { return log(x); }
    static __device__ __forceinline__ double exp_d(double x) { return exp(x); }
    static __device__ __forceinline__ double log1p_d(double x) { return log1p(x); }
    static __device__ __forceinline__ double abs_d(double x) { return fabs(x); }
    static __device__ __forceinline__ double sqrt_d(double x) { return sqrt(x); }
};

template<>
struct LogArithmeticTraits<float> {
    static constexpr float LOG_MIN = -87.0f;
    static constexpr float LOG_MAX = 88.0f;
    static constexpr float LOG_4PI_VAL = 2.53102425f;
    static constexpr float PI_VAL = 3.14159265f;

    // Use fast intrinsics for maximum performance
    static __device__ __forceinline__ float log_d(float x) { return __logf(x); }
    static __device__ __forceinline__ float exp_d(float x) { return __expf(x); }
    static __device__ __forceinline__ float log1p_d(float x) { return __logf(1.0f + x); }
    static __device__ __forceinline__ float abs_d(float x) { return fabsf(x); }
    static __device__ __forceinline__ float sqrt_d(float x) { return __fsqrt_rn(x); }
};

/**
 * logdiffexp: Computes log(a - b) given log(a) and log(b), assuming a > b
 *
 * Reference: jax_healpix/utils.py lines 14-20
 *
 * log(a-b) = log(a * (1 - b/a)) = log(a) + log(1 - exp(log_b - log_a))
 */
template<typename C>
__device__ __forceinline__
C logdiffexp_typed(C log_a, C log_b) {
    using Traits = LogArithmeticTraits<C>;
    C diff = log_b - log_a;
    return log_a + Traits::log1p_d(-Traits::exp_d(diff));
}

/**
 * logsumexp: Computes log(sign_A * A + sign_B * B) and resulting sign
 *
 * Reference: jax_healpix/utils.py lines 24-52
 *
 * This function computes the log of a sum of signed values in log-space,
 * which is critical for numerical stability in YLM recurrence.
 *
 * Given: A = sign_A * exp(log_A), B = sign_B * exp(log_B)
 * Computes: C = A + B = sign_result * exp(log_result)
 */
template<typename C>
__device__ __forceinline__
void logsumexp_typed(C log_A, C log_B,
                     int8_t sign_A, int8_t sign_B,
                     C* log_result, int8_t* sign_result) {
    using Traits = LogArithmeticTraits<C>;

    // Compute relative sign for the exponential term
    int relative_sign = sign_A * sign_B;

    // Find max and min for numerical stability
    C max_log, min_log;
    int8_t max_sign;
    if (log_A >= log_B) {
        max_log = log_A;
        min_log = log_B;
        max_sign = sign_A;
    } else {
        max_log = log_B;
        min_log = log_A;
        max_sign = sign_B;
    }

    // Core computation using log1p for stability
    // log(|result|) = max_log + log1p(relative_sign * exp(min_log - max_log))
    C delta = min_log - max_log;

    // Clamp to prevent underflow in exp
    if (delta < Traits::LOG_MIN) {
        // min_log is so small that it doesn't contribute
        *log_result = max_log;
        *sign_result = max_sign;
        return;
    }

    *log_result = max_log + Traits::log1p_d(relative_sign * Traits::exp_d(delta));

    // Determine output sign
    if (sign_A == sign_B) {
        // Same signs, result has that sign
        *sign_result = sign_A;
    } else if (log_A >= log_B) {
        // |A| >= |B|, result takes A's sign
        *sign_result = sign_A;
    } else {
        // |B| > |A|, result takes B's sign
        *sign_result = sign_B;
    }
}

/**
 * safe_log: Returns log(|x|), handling x == 0 gracefully
 * Returns LOG_MIN for x == 0
 */
template<typename C>
__device__ __forceinline__
C safe_log_typed(C x) {
    using Traits = LogArithmeticTraits<C>;
    C abs_x = Traits::abs_d(x);
    if (abs_x <= C(0)) {
        return Traits::LOG_MIN;
    }
    return Traits::log_d(abs_x);
}

/**
 * Clamp log values to prevent overflow on exp()
 */
template<typename C>
__device__ __forceinline__
C clamp_log_typed(C log_val) {
    using Traits = LogArithmeticTraits<C>;
    if (log_val < Traits::LOG_MIN) return Traits::LOG_MIN;
    if (log_val > Traits::LOG_MAX) return Traits::LOG_MAX;
    return log_val;
}

/**
 * log_A_lm: Compute log of recurrence coefficient A_{l,m}
 *
 * Reference: jax_healpix/YLM_jax_log.py lines 66-73
 * Eq. 14 of arXiv:1010.2084
 *
 * A_{l,m} = sqrt((4l^2 - 1) / (l^2 - m^2))
 * log(A) = 0.5 * [log(4l^2 - 1) - log(l^2 - m^2)]
 */
template<typename C>
__device__ __forceinline__
C log_A_lm(int l, int m) {
    using Traits = LogArithmeticTraits<C>;
    C l_c = C(l);
    C m_c = C(m);
    C l2 = l_c * l_c;
    C m2 = m_c * m_c;

    // log(4l^2 - 1)
    C log_num = Traits::log_d(C(4) * l2 - C(1));

    // log(l^2 - m^2)
    C log_denom = Traits::log_d(l2 - m2);

    return C(0.5) * (log_num - log_denom);
}

/**
 * log_B_lm: Compute log of recurrence coefficient B_{l,m} = A_{l,m} / A_{l-1,m}
 *
 * B_{l,m} = sqrt((2l+1)/(2l-3) * ((l-1)^2 - m^2) / (l^2 - m^2))
 */
template<typename C>
__device__ __forceinline__
C log_B_lm(int l, int m) {
    using Traits = LogArithmeticTraits<C>;
    C l_c = C(l);
    C m_c = C(m);
    C l2 = l_c * l_c;
    C m2 = m_c * m_c;
    C lm1_2 = C(l - 1) * C(l - 1);

    // (2l+1)/(2l-3) * ((l-1)^2 - m^2) / (l^2 - m^2)
    C num = (C(2) * l_c + C(1)) * (lm1_2 - m2);
    C denom = (C(2) * l_c - C(3)) * (l2 - m2);

    return C(0.5) * Traits::log_d(num / denom);
}

/**
 * Branchless logsumexp for Ylm recurrence: log(A - B) where A > 0, B > 0
 *
 * In Ylm recurrence: result = A*cos*Y_{l-1} - B*Y_{l-2}
 * We have: R1 = log|term1|, S1 = sign(term1), R2 = log|term2|, S2 = -sign(term2)
 *
 * This version avoids branches using select operations.
 */
template<typename C>
__device__ __forceinline__
void logsumexp_fast(C R1, C R2, int8_t S1, int8_t S2,
                    C* log_result, int8_t* sign_result) {
    using Traits = LogArithmeticTraits<C>;

    // Branchless max/min selection
    bool r1_larger = R1 >= R2;
    C max_log = r1_larger ? R1 : R2;
    C min_log = r1_larger ? R2 : R1;
    int8_t max_sign = r1_larger ? S1 : S2;

    C delta = min_log - max_log;
    int rel_sign = S1 * S2;

    // For very small delta, just return max
    // Use branchless: result = max_log + log1p(rel_sign * exp(delta)) if delta > LOG_MIN else max_log
    C exp_delta = Traits::exp_d(delta);
    C log1p_term = Traits::log_d(C(1) + C(rel_sign) * exp_delta);

    // Branchless select: if delta < LOG_MIN, use 0 for log1p_term
    bool use_log1p = delta >= Traits::LOG_MIN;
    *log_result = max_log + (use_log1p ? log1p_term : C(0));

    // Sign: if same signs -> that sign, else larger magnitude's sign
    *sign_result = (S1 == S2) ? S1 : max_sign;
}

/**
 * Compute cumulative log prefactor for Y[m,m] initialization
 *
 * Reference: jax_healpix/YLM_jax_log.py lines 39-42
 *
 * prefact[m] = prod_{j=1}^{m} sqrt((2j+1)/(2j))
 *
 * Using lgamma for O(1) computation instead of O(m) loop:
 * prefact = sqrt((2m+1)!! / (2m)!!) = sqrt((2m+1)! / (2^{2m} * (m!)^2))
 * log_prefact = 0.5 * [lgamma(2m+2) - 2m*log(2) - 2*lgamma(m+1)]
 */
template<typename C>
__device__ __forceinline__
C compute_log_prefact_ymm(int m) {
    using Traits = LogArithmeticTraits<C>;
    if (m == 0) return C(0);

    // Use lgamma for O(1) instead of O(m) loop
    C m_c = C(m);
    C lgamma_2m2 = lgamma(C(2) * m_c + C(2));  // lgamma(2m+2)
    C lgamma_m1 = lgamma(m_c + C(1));           // lgamma(m+1)
    C log2 = C(0.6931471805599453);             // log(2)

    return C(0.5) * (lgamma_2m2 - C(2) * m_c * log2 - C(2) * lgamma_m1);
}

// ============================================================================
// Legacy non-templated versions for backward compatibility
// ============================================================================

__device__ __forceinline__
real_t logdiffexp(real_t log_a, real_t log_b) {
    return logdiffexp_typed<real_t>(log_a, log_b);
}

__device__ __forceinline__
void logsumexp(real_t log_A, real_t log_B,
               int8_t sign_A, int8_t sign_B,
               real_t* log_result, int8_t* sign_result) {
    logsumexp_typed<real_t>(log_A, log_B, sign_A, sign_B, log_result, sign_result);
}

__device__ __forceinline__
real_t safe_log(real_t x) {
    return safe_log_typed<real_t>(x);
}

__device__ __forceinline__
real_t clamp_log(real_t log_val) {
    return clamp_log_typed<real_t>(log_val);
}

#endif // LOG_ARITHMETIC_CUH
