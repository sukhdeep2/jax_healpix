#ifndef LOG_ARITHMETIC_CUH
#define LOG_ARITHMETIC_CUH

#include "spht_types.h"
#include <math.h>

/**
 * logdiffexp: Computes log(a - b) given log(a) and log(b), assuming a > b
 *
 * Reference: jax_healpix/utils.py lines 14-20
 *
 * log(a-b) = log(a * (1 - b/a)) = log(a) + log(1 - exp(log_b - log_a))
 */
__device__ __forceinline__
real_t logdiffexp(real_t log_a, real_t log_b) {
    real_t diff = log_b - log_a;
    return log_a + log1p(-exp(diff));  // Use log1p for numerical stability
}

/**
 * logsumexp: Computes log(sign_A * A + sign_B * B) and resulting sign
 *
 * Reference: jax_healpix/utils.py lines 24-52
 *
 * This function computes the log of a sum of signed values in log-space,
 * which is critical for numerical stability in YLM recurrence.
 */
__device__ __forceinline__
void logsumexp(real_t log_A, real_t log_B,
               int8_t sign_A, int8_t sign_B,
               real_t* log_result, int8_t* sign_result) {
    // Compute relative sign
    int relative_sign = sign_A * sign_B;

    // Find max and min for numerical stability
    real_t max_log, min_log;
    if (log_A >= log_B) {
        max_log = log_A;
        min_log = log_B;
    } else {
        max_log = log_B;
        min_log = log_A;
    }

    // Core computation using log1p for stability
    // log(|result|) = max_log + log1p(relative_sign * exp(min_log - max_log))
    *log_result = max_log + log1p(relative_sign * exp(min_log - max_log));

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
 * safe_log: Returns log(x), handling x <= 0 gracefully
 * Returns -inf for x <= 0
 */
__device__ __forceinline__
real_t safe_log(real_t x) {
    if (x <= 0.0) {
        return -INFINITY;
    }
    return log(x);
}

/**
 * Clamp log values to prevent overflow on exp()
 */
__device__ __forceinline__
real_t clamp_log(real_t log_val) {
    if (log_val < LOG_MIN_F64) return LOG_MIN_F64;
    if (log_val > LOG_MAX_F64) return LOG_MAX_F64;
    return log_val;
}

#endif // LOG_ARITHMETIC_CUH
