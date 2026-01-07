#ifndef LOGSPACE_DFT_CUH
#define LOGSPACE_DFT_CUH

/**
 * Log-Space DFT & Bluestein Primitives
 *
 * Ring DFT in log-space for spherical harmonic transforms.
 *
 * DFT: F[m] = Σ_k f[k] * exp(-i * m * φ[k])
 *
 * In log-space:
 * - Input f[k] is (log_f, sign_f)
 * - Twiddle exp(-i*m*φ) = cos(m*φ) - i*sin(m*φ)
 * - Product: f * twiddle has real/imag parts that accumulate via logsumexp
 *
 * Output is Log-Polar (log_r, θ) which can be converted to Log-Cartesian
 * for Phase 2 operations.
 */

#include "logspace_complex.cuh"

// ============================================================================
// Twiddle Factor Computation
// ============================================================================

/**
 * LogTwiddle: Precomputed twiddle factor exp(-i*m*φ) in log-Cartesian
 *
 * exp(-i*θ) = cos(θ) - i*sin(θ)
 */
template<typename T>
struct LogTwiddle {
    T log_cos;       // log(|cos(m*φ)|)
    T log_sin;       // log(|sin(m*φ)|)
    int8_t sign_cos; // sign of cos(m*φ)
    int8_t sign_sin; // sign of sin(m*φ) (negated for exp(-i*θ))
};

/**
 * compute_log_twiddle: Compute exp(-i*m*φ) in log form
 *
 * exp(-i*m*φ) = cos(m*φ) - i*sin(m*φ)
 */
template<typename T>
__device__ __forceinline__
LogTwiddle<T> compute_log_twiddle(T m_phi) {
    using Traits = LogArithmeticTraits<T>;
    LogTwiddle<T> tw;

    T cos_mp = Traits::cos_d(m_phi);
    T sin_mp = Traits::sin_d(m_phi);

    T abs_cos = Traits::abs_d(cos_mp);
    T abs_sin = Traits::abs_d(sin_mp);

    tw.log_cos = (abs_cos > T(0)) ? Traits::log_d(abs_cos) : Traits::LOG_MIN;
    tw.log_sin = (abs_sin > T(0)) ? Traits::log_d(abs_sin) : Traits::LOG_MIN;
    tw.sign_cos = (cos_mp >= T(0)) ? int8_t(1) : int8_t(-1);
    // For exp(-i*θ), imaginary part is -sin(θ)
    tw.sign_sin = (sin_mp >= T(0)) ? int8_t(-1) : int8_t(1);

    return tw;
}

/**
 * compute_log_twiddle_conjugate: Compute exp(+i*m*φ) in log form
 *
 * exp(+i*m*φ) = cos(m*φ) + i*sin(m*φ)
 * Used for inverse DFT (alm2map).
 */
template<typename T>
__device__ __forceinline__
LogTwiddle<T> compute_log_twiddle_conjugate(T m_phi) {
    using Traits = LogArithmeticTraits<T>;
    LogTwiddle<T> tw;

    T cos_mp = Traits::cos_d(m_phi);
    T sin_mp = Traits::sin_d(m_phi);

    T abs_cos = Traits::abs_d(cos_mp);
    T abs_sin = Traits::abs_d(sin_mp);

    tw.log_cos = (abs_cos > T(0)) ? Traits::log_d(abs_cos) : Traits::LOG_MIN;
    tw.log_sin = (abs_sin > T(0)) ? Traits::log_d(abs_sin) : Traits::LOG_MIN;
    tw.sign_cos = (cos_mp >= T(0)) ? int8_t(1) : int8_t(-1);
    // For exp(+i*θ), imaginary part is +sin(θ)
    tw.sign_sin = (sin_mp >= T(0)) ? int8_t(1) : int8_t(-1);

    return tw;
}

// ============================================================================
// Ring DFT Accumulation
// ============================================================================

/**
 * log_dft_accumulate: Accumulate one ring pixel contribution to DFT
 *
 * F[m] += f[k] * exp(-i*m*φ[k])
 *       = f[k] * (cos(m*φ) - i*sin(m*φ))
 *       = f[k]*cos(m*φ) - i*f[k]*sin(m*φ)
 *
 * Real contribution: f * cos(m*φ)
 * Imag contribution: -f * sin(m*φ)
 *
 * @param log_f, sign_f: Input pixel value in log-space
 * @param tw: Precomputed twiddle factor
 * @param acc: Running accumulator (LogComplex)
 */
template<typename T>
__device__ __forceinline__
void log_dft_accumulate(T log_f, int8_t sign_f,
                        LogTwiddle<T> tw,
                        LogComplex<T>* acc) {
    // Real contribution: f * cos(m*φ)
    T log_re_contrib = log_f + tw.log_cos;
    int8_t sign_re_contrib = sign_f * tw.sign_cos;

    // Imag contribution: -f * sin(m*φ) (sign already negated in tw.sign_sin)
    T log_im_contrib = log_f + tw.log_sin;
    int8_t sign_im_contrib = sign_f * tw.sign_sin;

    // Accumulate using logsumexp
    logsumexp_typed<T>(acc->log_re, log_re_contrib, acc->sign_re, sign_re_contrib,
                       &acc->log_re, &acc->sign_re);
    logsumexp_typed<T>(acc->log_im, log_im_contrib, acc->sign_im, sign_im_contrib,
                       &acc->log_im, &acc->sign_im);
}

/**
 * log_idft_accumulate: Accumulate one m contribution to inverse DFT
 *
 * f[k] += F[m] * exp(+i*m*φ[k])
 *
 * Where F[m] is complex in log-Cartesian form.
 *
 * @param F_m: DFT coefficient (LogComplex)
 * @param tw: Precomputed twiddle factor for exp(+i*m*φ)
 * @param acc: Running accumulator for pixel value (LogComplex)
 */
template<typename T>
__device__ __forceinline__
void log_idft_accumulate(LogComplex<T> F_m,
                         LogTwiddle<T> tw,
                         LogComplex<T>* acc) {
    // F[m] * exp(+i*m*φ) = (F_re + i*F_im)(cos + i*sin)
    //                    = (F_re*cos - F_im*sin) + i*(F_re*sin + F_im*cos)

    // Real part: F_re*cos - F_im*sin
    T log_rc = F_m.log_re + tw.log_cos;
    T log_is = F_m.log_im + tw.log_sin;
    int8_t sign_rc = F_m.sign_re * tw.sign_cos;
    int8_t sign_is = F_m.sign_im * tw.sign_sin;

    T log_re_contrib;
    int8_t sign_re_contrib;
    logsumexp_typed<T>(log_rc, log_is, sign_rc, -sign_is,
                       &log_re_contrib, &sign_re_contrib);

    // Imag part: F_re*sin + F_im*cos
    T log_rs = F_m.log_re + tw.log_sin;
    T log_ic = F_m.log_im + tw.log_cos;
    int8_t sign_rs = F_m.sign_re * tw.sign_sin;
    int8_t sign_ic = F_m.sign_im * tw.sign_cos;

    T log_im_contrib;
    int8_t sign_im_contrib;
    logsumexp_typed<T>(log_rs, log_ic, sign_rs, sign_ic,
                       &log_im_contrib, &sign_im_contrib);

    // Accumulate
    logsumexp_typed<T>(acc->log_re, log_re_contrib, acc->sign_re, sign_re_contrib,
                       &acc->log_re, &acc->sign_re);
    logsumexp_typed<T>(acc->log_im, log_im_contrib, acc->sign_im, sign_im_contrib,
                       &acc->log_im, &acc->sign_im);
}

// ============================================================================
// Bluestein's Algorithm Support
// ============================================================================

/**
 * LogBluesteinChirp: Chirp factor W^(n²/2) for Bluestein's algorithm
 *
 * W = exp(-2πi/N), so W^(n²/2) = exp(-πi*n²/N)
 */
template<typename T>
struct LogBluesteinChirp {
    T log_re;       // log(|cos(π*n²/N)|)
    T log_im;       // log(|sin(π*n²/N)|)
    int8_t sign_re;
    int8_t sign_im;
};

/**
 * compute_log_bluestein_chirp: Compute W^(n²/2) = exp(-πi*n²/N)
 *
 * @param n: Sample index
 * @param N: DFT size
 */
template<typename T>
__device__ __forceinline__
LogBluesteinChirp<T> compute_log_bluestein_chirp(int n, int N) {
    using Traits = LogArithmeticTraits<T>;
    LogBluesteinChirp<T> chirp;

    // Phase = -π*n²/N
    T phase = -Traits::PI_VAL * T(n) * T(n) / T(N);
    T cos_p = Traits::cos_d(phase);
    T sin_p = Traits::sin_d(phase);

    T abs_cos = Traits::abs_d(cos_p);
    T abs_sin = Traits::abs_d(sin_p);

    chirp.log_re = (abs_cos > T(0)) ? Traits::log_d(abs_cos) : Traits::LOG_MIN;
    chirp.log_im = (abs_sin > T(0)) ? Traits::log_d(abs_sin) : Traits::LOG_MIN;
    chirp.sign_re = (cos_p >= T(0)) ? int8_t(1) : int8_t(-1);
    chirp.sign_im = (sin_p >= T(0)) ? int8_t(1) : int8_t(-1);

    return chirp;
}

/**
 * log_bluestein_premul: Multiply input by chirp before FFT
 *
 * For Bluestein: y[n] = x[n] * W^(n²/2)
 *
 * @param log_x, sign_x: Input value (real) in log-space
 * @param chirp: Precomputed chirp factor
 * @return Complex result in log-Cartesian form
 */
template<typename T>
__device__ __forceinline__
LogComplex<T> log_bluestein_premul(T log_x, int8_t sign_x,
                                   LogBluesteinChirp<T> chirp) {
    // x * (cos + i*sin) = x*cos + i*x*sin
    LogComplex<T> result;
    result.log_re = log_x + chirp.log_re;
    result.log_im = log_x + chirp.log_im;
    result.sign_re = sign_x * chirp.sign_re;
    result.sign_im = sign_x * chirp.sign_im;
    return result;
}

/**
 * log_bluestein_premul_complex: Multiply complex input by chirp
 *
 * @param z: Complex input in log-Cartesian form
 * @param chirp: Precomputed chirp factor
 * @return Complex result in log-Cartesian form
 */
template<typename T>
__device__ __forceinline__
LogComplex<T> log_bluestein_premul_complex(LogComplex<T> z,
                                           LogBluesteinChirp<T> chirp) {
    // (a + bi)(c + di) = (ac - bd) + i(ad + bc)
    LogComplex<T> result;

    // Real: ac - bd
    T log_ac = z.log_re + chirp.log_re;
    T log_bd = z.log_im + chirp.log_im;
    int8_t sign_ac = z.sign_re * chirp.sign_re;
    int8_t sign_bd = z.sign_im * chirp.sign_im;

    logsumexp_typed<T>(log_ac, log_bd, sign_ac, -sign_bd,
                       &result.log_re, &result.sign_re);

    // Imag: ad + bc
    T log_ad = z.log_re + chirp.log_im;
    T log_bc = z.log_im + chirp.log_re;
    int8_t sign_ad = z.sign_re * chirp.sign_im;
    int8_t sign_bc = z.sign_im * chirp.sign_re;

    logsumexp_typed<T>(log_ad, log_bc, sign_ad, sign_bc,
                       &result.log_im, &result.sign_im);

    return result;
}

/**
 * log_bluestein_postmul: Multiply FFT output by chirp conjugate
 *
 * For Bluestein: X[k] = W^(k²/2) * IFFT(FFT(y) * FFT(chirp))
 *
 * @param z: FFT output in log-Cartesian form
 * @param chirp: Chirp factor (will use conjugate)
 * @return Final DFT coefficient in log-Cartesian form
 */
template<typename T>
__device__ __forceinline__
LogComplex<T> log_bluestein_postmul(LogComplex<T> z,
                                    LogBluesteinChirp<T> chirp) {
    // Multiply by conjugate: (c - di)
    LogComplex<T> result;

    // Real: ac + bd (note: -(-bd) = +bd)
    T log_ac = z.log_re + chirp.log_re;
    T log_bd = z.log_im + chirp.log_im;
    int8_t sign_ac = z.sign_re * chirp.sign_re;
    int8_t sign_bd = z.sign_im * chirp.sign_im;

    logsumexp_typed<T>(log_ac, log_bd, sign_ac, sign_bd,
                       &result.log_re, &result.sign_re);

    // Imag: bc - ad
    T log_bc = z.log_im + chirp.log_re;
    T log_ad = z.log_re + chirp.log_im;
    int8_t sign_bc = z.sign_im * chirp.sign_re;
    int8_t sign_ad = z.sign_re * chirp.sign_im;

    logsumexp_typed<T>(log_bc, log_ad, sign_bc, -sign_ad,
                       &result.log_im, &result.sign_im);

    return result;
}

// ============================================================================
// Twiddle Recurrence (Angle Stepping)
// ============================================================================

/**
 * TwiddleRecurrence: Incremental twiddle factor computation
 *
 * Instead of computing cos(m*φ), sin(m*φ) from scratch each m,
 * use: exp(i*(m+1)*φ) = exp(i*m*φ) * exp(i*φ)
 *
 * Periodic resync recommended every 16-32 steps to prevent error accumulation.
 */
template<typename T>
struct TwiddleRecurrence {
    // Current twiddle exp(i*m*φ)
    T log_cos_m;
    T log_sin_m;
    int8_t sign_cos_m;
    int8_t sign_sin_m;

    // Base step exp(i*φ)
    T log_cos_1;
    T log_sin_1;
    int8_t sign_cos_1;
    int8_t sign_sin_1;

    // Initialize for m=0
    __device__ __forceinline__
    void init(T phi) {
        using Traits = LogArithmeticTraits<T>;

        // m=0: exp(i*0) = 1 + 0i
        log_cos_m = T(0);
        log_sin_m = Traits::LOG_MIN;
        sign_cos_m = int8_t(1);
        sign_sin_m = int8_t(0);

        // Base: exp(i*phi)
        T cos_phi = Traits::cos_d(phi);
        T sin_phi = Traits::sin_d(phi);

        log_cos_1 = safe_log_typed<T>(Traits::abs_d(cos_phi));
        log_sin_1 = safe_log_typed<T>(Traits::abs_d(sin_phi));
        sign_cos_1 = sign_of<T>(cos_phi);
        sign_sin_1 = sign_of<T>(sin_phi);
    }

    // Advance to m+1 using complex multiplication
    __device__ __forceinline__
    void step() {
        // exp(i*(m+1)*φ) = exp(i*m*φ) * exp(i*φ)
        // (cos_m + i*sin_m)(cos_1 + i*sin_1)
        // = (cos_m*cos_1 - sin_m*sin_1) + i*(cos_m*sin_1 + sin_m*cos_1)

        T new_log_cos, new_log_sin;
        int8_t new_sign_cos, new_sign_sin;

        // Real: cos_m*cos_1 - sin_m*sin_1
        T log_cc = log_cos_m + log_cos_1;
        T log_ss = log_sin_m + log_sin_1;
        logsumexp_typed<T>(log_cc, log_ss,
                           sign_cos_m * sign_cos_1,
                           -(sign_sin_m * sign_sin_1),
                           &new_log_cos, &new_sign_cos);

        // Imag: cos_m*sin_1 + sin_m*cos_1
        T log_cs = log_cos_m + log_sin_1;
        T log_sc = log_sin_m + log_cos_1;
        logsumexp_typed<T>(log_cs, log_sc,
                           sign_cos_m * sign_sin_1,
                           sign_sin_m * sign_cos_1,
                           &new_log_sin, &new_sign_sin);

        log_cos_m = new_log_cos;
        log_sin_m = new_log_sin;
        sign_cos_m = new_sign_cos;
        sign_sin_m = new_sign_sin;
    }

    // Resync to exact value (call periodically to prevent error accumulation)
    __device__ __forceinline__
    void resync(int m, T phi) {
        using Traits = LogArithmeticTraits<T>;
        T m_phi = T(m) * phi;
        T cos_mp = Traits::cos_d(m_phi);
        T sin_mp = Traits::sin_d(m_phi);

        log_cos_m = safe_log_typed<T>(Traits::abs_d(cos_mp));
        log_sin_m = safe_log_typed<T>(Traits::abs_d(sin_mp));
        sign_cos_m = sign_of<T>(cos_mp);
        sign_sin_m = sign_of<T>(sin_mp);
    }

    // Get current twiddle as LogTwiddle (for exp(-i*m*φ))
    __device__ __forceinline__
    LogTwiddle<T> get_twiddle() const {
        LogTwiddle<T> tw;
        tw.log_cos = log_cos_m;
        tw.log_sin = log_sin_m;
        tw.sign_cos = sign_cos_m;
        tw.sign_sin = -sign_sin_m;  // Negate for exp(-i*θ)
        return tw;
    }

    // Get current twiddle for exp(+i*m*φ) (inverse DFT)
    __device__ __forceinline__
    LogTwiddle<T> get_twiddle_conj() const {
        LogTwiddle<T> tw;
        tw.log_cos = log_cos_m;
        tw.log_sin = log_sin_m;
        tw.sign_cos = sign_cos_m;
        tw.sign_sin = sign_sin_m;  // Keep positive for exp(+i*θ)
        return tw;
    }
};

// Recommended resync interval
constexpr int TWIDDLE_RESYNC_INTERVAL = 16;

#endif // LOGSPACE_DFT_CUH
