#ifndef LOGSPACE_COMPLEX_CUH
#define LOGSPACE_COMPLEX_CUH

/**
 * Log-Space Complex Arithmetic
 *
 * Complex numbers in log-Cartesian representation:
 *   z = (sign_re * exp(log_re)) + i * (sign_im * exp(log_im))
 *
 * This representation allows:
 * - Very large/small magnitudes without overflow
 * - Efficient multiply (just add logs)
 * - Add via logsumexp (more expensive but stable)
 *
 * Used for Phase 2 (Ylm accumulation) after conversion from Log-Polar DFT output.
 */

#include "logspace_math.cuh"

// ============================================================================
// Log-Complex Types
// ============================================================================

/**
 * LogComplex: Complex number in log-Cartesian representation
 *
 * Value = sign_re * exp(log_re) + i * sign_im * exp(log_im)
 */
template<typename T>
struct LogComplex {
    T log_re;         // log(|real part|)
    T log_im;         // log(|imag part|)
    int8_t sign_re;   // sign of real part: +1, -1, or 0
    int8_t sign_im;   // sign of imag part: +1, -1, or 0

    // Initialize to zero
    __device__ __forceinline__
    static LogComplex<T> zero() {
        LogComplex<T> z;
        z.log_re = LogArithmeticTraits<T>::LOG_MIN;
        z.log_im = LogArithmeticTraits<T>::LOG_MIN;
        z.sign_re = 0;
        z.sign_im = 0;
        return z;
    }

    // Initialize from linear complex
    __device__ __forceinline__
    static LogComplex<T> from_linear(T re, T im) {
        LogComplex<T> z;
        z.log_re = safe_log_typed<T>(re);
        z.log_im = safe_log_typed<T>(im);
        z.sign_re = sign_of<T>(re);
        z.sign_im = sign_of<T>(im);
        return z;
    }

    // Convert to linear complex
    __device__ __forceinline__
    void to_linear(T* re, T* im) const {
        using Traits = LogArithmeticTraits<T>;
        *re = sign_re * Traits::exp_d(clamp_log_typed<T>(log_re));
        *im = sign_im * Traits::exp_d(clamp_log_typed<T>(log_im));
    }
};

/**
 * LogPolar: Complex number in log-polar representation
 *
 * Value = exp(log_r) * exp(i * theta) = exp(log_r) * (cos(theta) + i*sin(theta))
 *
 * Natural output from DFT (magnitude + phase).
 */
template<typename T>
struct LogPolar {
    T log_r;   // log(magnitude)
    T theta;   // phase angle

    // Initialize to zero
    __device__ __forceinline__
    static LogPolar<T> zero() {
        LogPolar<T> z;
        z.log_r = LogArithmeticTraits<T>::LOG_MIN;
        z.theta = T(0);
        return z;
    }
};

// ============================================================================
// Conversions
// ============================================================================

/**
 * Convert Log-Polar to Log-Cartesian
 *
 * From: r * exp(i*theta) = r*cos(theta) + i*r*sin(theta)
 * To: sign_re * exp(log_re) + i * sign_im * exp(log_im)
 */
template<typename T>
__device__ __forceinline__
LogComplex<T> log_polar_to_cartesian(LogPolar<T> polar) {
    using Traits = LogArithmeticTraits<T>;
    LogComplex<T> cart;

    T cos_th = Traits::cos_d(polar.theta);
    T sin_th = Traits::sin_d(polar.theta);

    // Handle cos/sin = 0 gracefully
    T abs_cos = Traits::abs_d(cos_th);
    T abs_sin = Traits::abs_d(sin_th);

    cart.log_re = (abs_cos > T(0)) ? polar.log_r + Traits::log_d(abs_cos) : Traits::LOG_MIN;
    cart.log_im = (abs_sin > T(0)) ? polar.log_r + Traits::log_d(abs_sin) : Traits::LOG_MIN;
    cart.sign_re = (cos_th >= T(0)) ? int8_t(1) : int8_t(-1);
    cart.sign_im = (sin_th >= T(0)) ? int8_t(1) : int8_t(-1);

    return cart;
}

/**
 * Convert Log-Cartesian to Log-Polar
 *
 * From: sign_re * exp(log_re) + i * sign_im * exp(log_im)
 * To: exp(log_r) * exp(i * theta)
 *
 * Note: This involves computing magnitude and atan2, which is expensive.
 * Prefer staying in Cartesian for Phase 2 operations.
 */
template<typename T>
__device__ __forceinline__
LogPolar<T> log_cartesian_to_polar(LogComplex<T> cart) {
    using Traits = LogArithmeticTraits<T>;
    LogPolar<T> polar;

    // Convert to linear for atan2
    T re = cart.sign_re * Traits::exp_d(clamp_log_typed<T>(cart.log_re));
    T im = cart.sign_im * Traits::exp_d(clamp_log_typed<T>(cart.log_im));

    // Magnitude: |z| = sqrt(re^2 + im^2)
    // In log-space: log|z| = 0.5 * log(re^2 + im^2)
    // Using logsumexp: log(|re|^2 + |im|^2) where both terms are positive
    T log_re2 = T(2) * cart.log_re;
    T log_im2 = T(2) * cart.log_im;

    // Both terms are positive squares, so signs are +1
    T log_sum;
    int8_t dummy_sign;
    logsumexp_typed<T>(log_re2, log_im2, int8_t(1), int8_t(1), &log_sum, &dummy_sign);
    polar.log_r = T(0.5) * log_sum;

    // Phase
    polar.theta = atan2(im, re);

    return polar;
}

// ============================================================================
// Log-Complex Arithmetic
// ============================================================================

/**
 * log_complex_add: z1 + z2 in log-Cartesian representation
 *
 * Adds real and imaginary parts separately using logsumexp.
 */
template<typename T>
__device__ __forceinline__
LogComplex<T> log_complex_add(LogComplex<T> z1, LogComplex<T> z2) {
    LogComplex<T> result;

    logsumexp_typed<T>(z1.log_re, z2.log_re, z1.sign_re, z2.sign_re,
                       &result.log_re, &result.sign_re);
    logsumexp_typed<T>(z1.log_im, z2.log_im, z1.sign_im, z2.sign_im,
                       &result.log_im, &result.sign_im);

    return result;
}

/**
 * log_complex_mul: z1 * z2 in log-Cartesian representation
 *
 * (a + bi)(c + di) = (ac - bd) + i(ad + bc)
 *
 * In log-space:
 * - ac: log(|ac|) = log_a + log_c, sign = sign_a * sign_c
 * - bd: log(|bd|) = log_b + log_d, sign = sign_b * sign_d
 * - Real = ac - bd: use logsumexp with negated sign for bd
 * - ad, bc similarly for imaginary part
 */
template<typename T>
__device__ __forceinline__
LogComplex<T> log_complex_mul(LogComplex<T> z1, LogComplex<T> z2) {
    LogComplex<T> result;

    // Real part: ac - bd
    T log_ac = z1.log_re + z2.log_re;
    T log_bd = z1.log_im + z2.log_im;
    int8_t sign_ac = z1.sign_re * z2.sign_re;
    int8_t sign_bd = z1.sign_im * z2.sign_im;

    logsumexp_typed<T>(log_ac, log_bd, sign_ac, -sign_bd,
                       &result.log_re, &result.sign_re);

    // Imag part: ad + bc
    T log_ad = z1.log_re + z2.log_im;
    T log_bc = z1.log_im + z2.log_re;
    int8_t sign_ad = z1.sign_re * z2.sign_im;
    int8_t sign_bc = z1.sign_im * z2.sign_re;

    logsumexp_typed<T>(log_ad, log_bc, sign_ad, sign_bc,
                       &result.log_im, &result.sign_im);

    return result;
}

/**
 * log_complex_mul_real: z * r where r is a real value in log-space
 *
 * (a + bi) * r = ar + i*br
 * In log-space: just add log_r to both components
 */
template<typename T>
__device__ __forceinline__
LogComplex<T> log_complex_mul_real(LogComplex<T> z, T log_r, int8_t sign_r) {
    LogComplex<T> result;
    result.log_re = z.log_re + log_r;
    result.log_im = z.log_im + log_r;
    result.sign_re = z.sign_re * sign_r;
    result.sign_im = z.sign_im * sign_r;
    return result;
}

/**
 * log_complex_conj: Conjugate of z
 *
 * conj(a + bi) = a - bi
 * Just negate sign_im
 */
template<typename T>
__device__ __forceinline__
LogComplex<T> log_complex_conj(LogComplex<T> z) {
    LogComplex<T> result = z;
    result.sign_im = -z.sign_im;
    return result;
}

/**
 * log_complex_from_angle: Create exp(i*theta) in log-Cartesian form
 *
 * exp(i*theta) = cos(theta) + i*sin(theta)
 */
template<typename T>
__device__ __forceinline__
LogComplex<T> log_complex_from_angle(T theta) {
    using Traits = LogArithmeticTraits<T>;
    LogComplex<T> result;

    T cos_th = Traits::cos_d(theta);
    T sin_th = Traits::sin_d(theta);

    result.log_re = safe_log_typed<T>(Traits::abs_d(cos_th));
    result.log_im = safe_log_typed<T>(Traits::abs_d(sin_th));
    result.sign_re = (cos_th >= T(0)) ? int8_t(1) : int8_t(-1);
    result.sign_im = (sin_th >= T(0)) ? int8_t(1) : int8_t(-1);

    return result;
}

/**
 * log_complex_scale_angle: Create r * exp(i*theta) where r is in log-space
 *
 * r * exp(i*theta) = r*cos(theta) + i*r*sin(theta)
 */
template<typename T>
__device__ __forceinline__
LogComplex<T> log_complex_scale_angle(T log_r, int8_t sign_r, T theta) {
    using Traits = LogArithmeticTraits<T>;
    LogComplex<T> result;

    T cos_th = Traits::cos_d(theta);
    T sin_th = Traits::sin_d(theta);

    T abs_cos = Traits::abs_d(cos_th);
    T abs_sin = Traits::abs_d(sin_th);

    result.log_re = (abs_cos > T(0)) ? log_r + Traits::log_d(abs_cos) : Traits::LOG_MIN;
    result.log_im = (abs_sin > T(0)) ? log_r + Traits::log_d(abs_sin) : Traits::LOG_MIN;
    result.sign_re = sign_r * ((cos_th >= T(0)) ? int8_t(1) : int8_t(-1));
    result.sign_im = sign_r * ((sin_th >= T(0)) ? int8_t(1) : int8_t(-1));

    return result;
}

// ============================================================================
// Accumulation Helpers (for Ylm synthesis/analysis)
// ============================================================================

/**
 * log_complex_accumulate: acc += contrib in log-Cartesian space
 *
 * This is the core operation for Phase 2 accumulation.
 */
template<typename T>
__device__ __forceinline__
void log_complex_accumulate(LogComplex<T>* acc, LogComplex<T> contrib) {
    logsumexp_typed<T>(acc->log_re, contrib.log_re, acc->sign_re, contrib.sign_re,
                       &acc->log_re, &acc->sign_re);
    logsumexp_typed<T>(acc->log_im, contrib.log_im, acc->sign_im, contrib.sign_im,
                       &acc->log_im, &acc->sign_im);
}

/**
 * log_complex_accumulate_scaled: acc += factor * contrib
 *
 * Where factor is a real value in log-space (log_f, sign_f).
 * Common pattern: acc += Y_lm * (F_m * w) for map2alm
 */
template<typename T>
__device__ __forceinline__
void log_complex_accumulate_scaled(LogComplex<T>* acc,
                                   LogComplex<T> contrib,
                                   T log_f, int8_t sign_f) {
    // Scale contribution
    T log_re_scaled = contrib.log_re + log_f;
    T log_im_scaled = contrib.log_im + log_f;
    int8_t sign_re_scaled = contrib.sign_re * sign_f;
    int8_t sign_im_scaled = contrib.sign_im * sign_f;

    // Accumulate
    logsumexp_typed<T>(acc->log_re, log_re_scaled, acc->sign_re, sign_re_scaled,
                       &acc->log_re, &acc->sign_re);
    logsumexp_typed<T>(acc->log_im, log_im_scaled, acc->sign_im, sign_im_scaled,
                       &acc->log_im, &acc->sign_im);
}

#endif // LOGSPACE_COMPLEX_CUH
