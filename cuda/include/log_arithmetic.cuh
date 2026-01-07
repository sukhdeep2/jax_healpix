#ifndef LOG_ARITHMETIC_CUH
#define LOG_ARITHMETIC_CUH

/**
 * Log-Arithmetic Master Header
 *
 * This header provides backward compatibility by including all log-space
 * arithmetic modules and providing legacy (non-templated) wrappers.
 *
 * New code should include specific modules:
 * - logspace_math.cuh    : Core real arithmetic (logsumexp, safe_log, etc.)
 * - logspace_complex.cuh : Complex arithmetic (LogComplex, add, mul)
 * - logspace_dft.cuh     : Ring DFT & Bluestein primitives
 *
 * Legacy code can continue using this header with non-templated wrappers.
 */

#include "spht_types.h"
#include "logspace_math.cuh"
#include "logspace_complex.cuh"
#include "logspace_dft.cuh"

// ============================================================================
// Legacy Non-Templated Wrappers (for backward compatibility)
// ============================================================================

/**
 * logdiffexp: Legacy wrapper using real_t (typically double)
 */
__device__ __forceinline__
real_t logdiffexp(real_t log_a, real_t log_b) {
    return logdiffexp_typed<real_t>(log_a, log_b);
}

/**
 * logsumexp: Legacy wrapper using real_t
 */
__device__ __forceinline__
void logsumexp(real_t log_A, real_t log_B,
               int8_t sign_A, int8_t sign_B,
               real_t* log_result, int8_t* sign_result) {
    logsumexp_typed<real_t>(log_A, log_B, sign_A, sign_B, log_result, sign_result);
}

/**
 * safe_log: Legacy wrapper using real_t
 */
__device__ __forceinline__
real_t safe_log(real_t x) {
    return safe_log_typed<real_t>(x);
}

/**
 * clamp_log: Legacy wrapper using real_t
 */
__device__ __forceinline__
real_t clamp_log(real_t log_val) {
    return clamp_log_typed<real_t>(log_val);
}

#endif // LOG_ARITHMETIC_CUH
