/**
 * Beam and ALM Operations
 *
 * Implements:
 * - gauss_beam: Compute Gaussian beam window function
 * - smoothalm: Apply beam smoothing to alm coefficients
 * - almxfl: Multiply alm by arbitrary f(l) filter
 * - pixwin: Approximate pixel window function
 * - resize_alm: Truncate or zero-pad alm arrays
 *
 * Reference: healpy documentation and IMPLEMENTATION_PLAN.md
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// Gaussian Beam Window Function
// ============================================================================

/**
 * Compute Gaussian beam window function B_l
 *
 * B_l = exp(-l(l+1) * σ² / 2)
 * σ = FWHM / sqrt(8 * ln(2))
 *
 * Also outputs log(B_l) for downstream log-space operations.
 */
template<typename T>
__global__ void gauss_beam_kernel(
    int l_max,
    T sigma_sq,                    // σ² = (FWHM / sqrt(8*ln2))²
    T* __restrict__ d_bl,          // [l_max+1] output beam
    T* __restrict__ d_log_bl       // [l_max+1] log(beam) for downstream
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l > l_max) return;

    T l_f = T(l);
    T log_bl = -l_f * (l_f + T(1)) * sigma_sq * T(0.5);

    d_log_bl[l] = log_bl;

    // Only compute exp for moderate values to avoid underflow
    if (log_bl > T(-700)) {
        d_bl[l] = exp(log_bl);
    } else {
        d_bl[l] = T(0);
    }
}

// ============================================================================
// Smooth ALM with Gaussian Beam
// ============================================================================

/**
 * Apply Gaussian beam smoothing to alm coefficients
 *
 * a_lm' = a_lm * B_l
 *
 * Uses log-space for very small beam values to maintain precision.
 */
template<typename T>
__global__ void smoothalm_kernel(
    int l_max,
    int n_fields,
    T sigma_sq,                            // σ² = (FWHM / sqrt(8*ln2))²
    const T* __restrict__ d_alm_real_in,
    const T* __restrict__ d_alm_imag_in,
    T* __restrict__ d_alm_real_out,
    T* __restrict__ d_alm_imag_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;

    if (idx >= total) return;

    int field = idx / (n_l * n_l);
    int lm_idx = idx % (n_l * n_l);
    int l = lm_idx / n_l;
    int m = lm_idx % n_l;

    if (m > l) return;  // Upper triangular only

    // Compute log(B_l) for numerical stability
    T l_f = T(l);
    T log_beam = -l_f * (l_f + T(1)) * sigma_sq * T(0.5);

    // Get input alm
    T alm_re = d_alm_real_in[idx];
    T alm_im = d_alm_imag_in[idx];

    // Apply beam in log-space if |log_beam| is large
    if (log_beam < T(-20)) {
        // Very small beam - use log-space
        T alm_mag_sq = alm_re * alm_re + alm_im * alm_im;
        if (alm_mag_sq > T(1e-35)) {
            T log_alm_mag = T(0.5) * log(alm_mag_sq);
            T phase = atan2(alm_im, alm_re);

            T log_result = log_alm_mag + log_beam;
            T result_mag = exp(log_result);

            d_alm_real_out[idx] = result_mag * cos(phase);
            d_alm_imag_out[idx] = result_mag * sin(phase);
        } else {
            d_alm_real_out[idx] = T(0);
            d_alm_imag_out[idx] = T(0);
        }
    } else {
        // Normal beam - direct multiplication
        T beam = exp(log_beam);
        d_alm_real_out[idx] = alm_re * beam;
        d_alm_imag_out[idx] = alm_im * beam;
    }
}

/**
 * In-place version of smoothalm
 */
template<typename T>
__global__ void smoothalm_inplace_kernel(
    int l_max,
    int n_fields,
    T sigma_sq,
    T* __restrict__ d_alm_real,
    T* __restrict__ d_alm_imag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;

    if (idx >= total) return;

    int lm_idx = idx % (n_l * n_l);
    int l = lm_idx / n_l;
    int m = lm_idx % n_l;

    if (m > l) return;

    T l_f = T(l);
    T log_beam = -l_f * (l_f + T(1)) * sigma_sq * T(0.5);

    T alm_re = d_alm_real[idx];
    T alm_im = d_alm_imag[idx];

    if (log_beam < T(-20)) {
        T alm_mag_sq = alm_re * alm_re + alm_im * alm_im;
        if (alm_mag_sq > T(1e-35)) {
            T log_alm_mag = T(0.5) * log(alm_mag_sq);
            T phase = atan2(alm_im, alm_re);
            T log_result = log_alm_mag + log_beam;
            T result_mag = exp(log_result);
            d_alm_real[idx] = result_mag * cos(phase);
            d_alm_imag[idx] = result_mag * sin(phase);
        } else {
            d_alm_real[idx] = T(0);
            d_alm_imag[idx] = T(0);
        }
    } else {
        T beam = exp(log_beam);
        d_alm_real[idx] = alm_re * beam;
        d_alm_imag[idx] = alm_im * beam;
    }
}

// ============================================================================
// ALM × f(l) Filter
// ============================================================================

/**
 * Multiply alm by arbitrary filter function f(l)
 *
 * a_lm' = a_lm * f_l
 *
 * Supports both linear and log-space computation for numerical stability.
 */
template<typename T>
__global__ void almxfl_kernel(
    int l_max,
    int n_fields,
    const T* __restrict__ d_fl,            // [l_max+1] filter function
    const T* __restrict__ d_log_fl,        // [l_max+1] log(|f_l|) for log-space
    const int8_t* __restrict__ d_sign_fl,  // [l_max+1] sign of f_l
    const T* __restrict__ d_alm_real_in,
    const T* __restrict__ d_alm_imag_in,
    T* __restrict__ d_alm_real_out,
    T* __restrict__ d_alm_imag_out,
    bool use_log_space
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;

    if (idx >= total) return;

    int lm_idx = idx % (n_l * n_l);
    int l = lm_idx / n_l;
    int m = lm_idx % n_l;

    if (m > l) return;

    T alm_re = d_alm_real_in[idx];
    T alm_im = d_alm_imag_in[idx];

    if (use_log_space) {
        T log_fl = d_log_fl[l];
        int8_t sign = d_sign_fl[l];

        T alm_mag_sq = alm_re * alm_re + alm_im * alm_im;
        if (alm_mag_sq > T(1e-35)) {
            T log_alm_mag = T(0.5) * log(alm_mag_sq);
            T phase = atan2(alm_im, alm_re);

            T log_result = log_alm_mag + log_fl;
            T result_mag = exp(log_result);

            d_alm_real_out[idx] = T(sign) * result_mag * cos(phase);
            d_alm_imag_out[idx] = T(sign) * result_mag * sin(phase);
        } else {
            d_alm_real_out[idx] = T(0);
            d_alm_imag_out[idx] = T(0);
        }
    } else {
        T fl = d_fl[l];
        d_alm_real_out[idx] = alm_re * fl;
        d_alm_imag_out[idx] = alm_im * fl;
    }
}

/**
 * In-place almxfl (simpler version, no log-space)
 */
template<typename T>
__global__ void almxfl_inplace_kernel(
    int l_max,
    int n_fields,
    const T* __restrict__ d_fl,
    T* __restrict__ d_alm_real,
    T* __restrict__ d_alm_imag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;

    if (idx >= total) return;

    int lm_idx = idx % (n_l * n_l);
    int l = lm_idx / n_l;
    int m = lm_idx % n_l;

    if (m > l) return;

    T fl = d_fl[l];
    d_alm_real[idx] *= fl;
    d_alm_imag[idx] *= fl;
}

// ============================================================================
// Pixel Window Function
// ============================================================================

/**
 * Compute approximate pixel window function
 *
 * W_l ≈ exp(-l² * θ_pix² / 2)
 * where θ_pix ≈ sqrt(4π / (12 * nside²))
 *
 * Note: For exact pixel windows, use precomputed HEALPix tables.
 */
template<typename T>
__global__ void pixwin_approx_kernel(
    int l_max,
    int nside,
    T* __restrict__ d_pixwin,
    T* __restrict__ d_log_pixwin
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l > l_max) return;

    // Approximate pixel window
    T theta_pix_sq = T(4.0 * M_PI) / (T(12) * T(nside) * T(nside));
    T l_f = T(l);
    T log_win = -l_f * l_f * theta_pix_sq * T(0.5);

    d_log_pixwin[l] = log_win;
    d_pixwin[l] = (log_win > T(-700)) ? exp(log_win) : T(0);
}

// ============================================================================
// Resize ALM
// ============================================================================

/**
 * Resize alm to different lmax (truncate or zero-pad)
 */
template<typename T>
__global__ void resize_alm_kernel(
    int l_max_in,
    int l_max_out,
    int n_fields,
    const T* __restrict__ d_alm_real_in,
    const T* __restrict__ d_alm_imag_in,
    T* __restrict__ d_alm_real_out,
    T* __restrict__ d_alm_imag_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_l_out = l_max_out + 1;
    int total = n_fields * n_l_out * n_l_out;

    if (idx >= total) return;

    int field = idx / (n_l_out * n_l_out);
    int lm_idx = idx % (n_l_out * n_l_out);
    int l = lm_idx / n_l_out;
    int m = lm_idx % n_l_out;

    if (m > l) return;

    int n_l_in = l_max_in + 1;

    if (l <= l_max_in && m <= l) {
        // Copy from input
        int in_idx = field * n_l_in * n_l_in + l * n_l_in + m;
        d_alm_real_out[idx] = d_alm_real_in[in_idx];
        d_alm_imag_out[idx] = d_alm_imag_in[in_idx];
    } else {
        // Zero pad
        d_alm_real_out[idx] = T(0);
        d_alm_imag_out[idx] = T(0);
    }
}

// ============================================================================
// C API Entry Points
// ============================================================================

extern "C" {

// Gaussian beam - f64
void gauss_beam_cuda_f64(int l_max, double fwhm_radians,
                          double* d_bl, double* d_log_bl) {
    // σ = FWHM / sqrt(8 * ln(2))
    double sigma = fwhm_radians / sqrt(8.0 * log(2.0));
    double sigma_sq = sigma * sigma;

    int threads = 256;
    int blocks = (l_max + 1 + threads - 1) / threads;
    gauss_beam_kernel<double><<<blocks, threads>>>(l_max, sigma_sq, d_bl, d_log_bl);
}

// Gaussian beam - f32
void gauss_beam_cuda_f32(int l_max, float fwhm_radians,
                          float* d_bl, float* d_log_bl) {
    float sigma = fwhm_radians / sqrtf(8.0f * logf(2.0f));
    float sigma_sq = sigma * sigma;

    int threads = 256;
    int blocks = (l_max + 1 + threads - 1) / threads;
    gauss_beam_kernel<float><<<blocks, threads>>>(l_max, sigma_sq, d_bl, d_log_bl);
}

// Smoothalm - f64
void smoothalm_cuda_f64(int l_max, int n_fields, double fwhm_radians,
                         const double* d_alm_real_in, const double* d_alm_imag_in,
                         double* d_alm_real_out, double* d_alm_imag_out) {
    double sigma = fwhm_radians / sqrt(8.0 * log(2.0));
    double sigma_sq = sigma * sigma;

    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    smoothalm_kernel<double><<<blocks, threads>>>(
        l_max, n_fields, sigma_sq,
        d_alm_real_in, d_alm_imag_in,
        d_alm_real_out, d_alm_imag_out
    );
}

// Smoothalm - f32
void smoothalm_cuda_f32(int l_max, int n_fields, float fwhm_radians,
                         const float* d_alm_real_in, const float* d_alm_imag_in,
                         float* d_alm_real_out, float* d_alm_imag_out) {
    float sigma = fwhm_radians / sqrtf(8.0f * logf(2.0f));
    float sigma_sq = sigma * sigma;

    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    smoothalm_kernel<float><<<blocks, threads>>>(
        l_max, n_fields, sigma_sq,
        d_alm_real_in, d_alm_imag_in,
        d_alm_real_out, d_alm_imag_out
    );
}

// Smoothalm inplace - f64
void smoothalm_inplace_cuda_f64(int l_max, int n_fields, double fwhm_radians,
                                 double* d_alm_real, double* d_alm_imag) {
    double sigma = fwhm_radians / sqrt(8.0 * log(2.0));
    double sigma_sq = sigma * sigma;

    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    smoothalm_inplace_kernel<double><<<blocks, threads>>>(
        l_max, n_fields, sigma_sq, d_alm_real, d_alm_imag
    );
}

// Smoothalm inplace - f32
void smoothalm_inplace_cuda_f32(int l_max, int n_fields, float fwhm_radians,
                                 float* d_alm_real, float* d_alm_imag) {
    float sigma = fwhm_radians / sqrtf(8.0f * logf(2.0f));
    float sigma_sq = sigma * sigma;

    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    smoothalm_inplace_kernel<float><<<blocks, threads>>>(
        l_max, n_fields, sigma_sq, d_alm_real, d_alm_imag
    );
}

// ALMxFL - f64
void almxfl_cuda_f64(int l_max, int n_fields,
                      const double* d_fl, const double* d_log_fl, const int8_t* d_sign_fl,
                      const double* d_alm_real_in, const double* d_alm_imag_in,
                      double* d_alm_real_out, double* d_alm_imag_out,
                      bool use_log_space) {
    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    almxfl_kernel<double><<<blocks, threads>>>(
        l_max, n_fields, d_fl, d_log_fl, d_sign_fl,
        d_alm_real_in, d_alm_imag_in,
        d_alm_real_out, d_alm_imag_out,
        use_log_space
    );
}

// ALMxFL - f32
void almxfl_cuda_f32(int l_max, int n_fields,
                      const float* d_fl, const float* d_log_fl, const int8_t* d_sign_fl,
                      const float* d_alm_real_in, const float* d_alm_imag_in,
                      float* d_alm_real_out, float* d_alm_imag_out,
                      bool use_log_space) {
    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    almxfl_kernel<float><<<blocks, threads>>>(
        l_max, n_fields, d_fl, d_log_fl, d_sign_fl,
        d_alm_real_in, d_alm_imag_in,
        d_alm_real_out, d_alm_imag_out,
        use_log_space
    );
}

// ALMxFL inplace (simple) - f64
void almxfl_inplace_cuda_f64(int l_max, int n_fields, const double* d_fl,
                              double* d_alm_real, double* d_alm_imag) {
    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    almxfl_inplace_kernel<double><<<blocks, threads>>>(
        l_max, n_fields, d_fl, d_alm_real, d_alm_imag
    );
}

// ALMxFL inplace (simple) - f32
void almxfl_inplace_cuda_f32(int l_max, int n_fields, const float* d_fl,
                              float* d_alm_real, float* d_alm_imag) {
    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    almxfl_inplace_kernel<float><<<blocks, threads>>>(
        l_max, n_fields, d_fl, d_alm_real, d_alm_imag
    );
}

// Pixwin - f64
void pixwin_cuda_f64(int l_max, int nside,
                      double* d_pixwin, double* d_log_pixwin) {
    int threads = 256;
    int blocks = (l_max + 1 + threads - 1) / threads;
    pixwin_approx_kernel<double><<<blocks, threads>>>(l_max, nside, d_pixwin, d_log_pixwin);
}

// Pixwin - f32
void pixwin_cuda_f32(int l_max, int nside,
                      float* d_pixwin, float* d_log_pixwin) {
    int threads = 256;
    int blocks = (l_max + 1 + threads - 1) / threads;
    pixwin_approx_kernel<float><<<blocks, threads>>>(l_max, nside, d_pixwin, d_log_pixwin);
}

// Resize ALM - f64
void resize_alm_cuda_f64(int l_max_in, int l_max_out, int n_fields,
                          const double* d_alm_real_in, const double* d_alm_imag_in,
                          double* d_alm_real_out, double* d_alm_imag_out) {
    int n_l_out = l_max_out + 1;
    int total = n_fields * n_l_out * n_l_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    resize_alm_kernel<double><<<blocks, threads>>>(
        l_max_in, l_max_out, n_fields,
        d_alm_real_in, d_alm_imag_in,
        d_alm_real_out, d_alm_imag_out
    );
}

// Resize ALM - f32
void resize_alm_cuda_f32(int l_max_in, int l_max_out, int n_fields,
                          const float* d_alm_real_in, const float* d_alm_imag_in,
                          float* d_alm_real_out, float* d_alm_imag_out) {
    int n_l_out = l_max_out + 1;
    int total = n_fields * n_l_out * n_l_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    resize_alm_kernel<float><<<blocks, threads>>>(
        l_max_in, l_max_out, n_fields,
        d_alm_real_in, d_alm_imag_in,
        d_alm_real_out, d_alm_imag_out
    );
}

} // extern "C"
