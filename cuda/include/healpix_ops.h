/**
 * HEALPix Operations Header
 *
 * C API for beam, alm filtering, synalm, and related operations.
 */

#ifndef HEALPIX_OPS_H
#define HEALPIX_OPS_H

#include <stdint.h>
#include <stdbool.h>

// Forward declaration for curandState (defined in curand_kernel.h)
struct curandStateXORWOW;
typedef struct curandStateXORWOW curandState;

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Gaussian Beam Window Function
// ============================================================================

/**
 * Compute Gaussian beam window function B_l
 *
 * @param l_max Maximum multipole
 * @param fwhm_radians Full-width half-maximum in radians
 * @param d_bl Output beam values [l_max+1] (device)
 * @param d_log_bl Output log(beam) values [l_max+1] (device)
 */
void gauss_beam_cuda_f64(int l_max, double fwhm_radians,
                          double* d_bl, double* d_log_bl);
void gauss_beam_cuda_f32(int l_max, float fwhm_radians,
                          float* d_bl, float* d_log_bl);

// ============================================================================
// Smooth ALM with Gaussian Beam
// ============================================================================

/**
 * Apply Gaussian beam smoothing to alm coefficients
 *
 * @param l_max Maximum multipole
 * @param n_fields Number of fields
 * @param fwhm_radians Beam FWHM in radians
 * @param d_alm_real_in Input real part [n_fields, l_max+1, l_max+1] (device)
 * @param d_alm_imag_in Input imag part (device)
 * @param d_alm_real_out Output real part (device)
 * @param d_alm_imag_out Output imag part (device)
 */
void smoothalm_cuda_f64(int l_max, int n_fields, double fwhm_radians,
                         const double* d_alm_real_in, const double* d_alm_imag_in,
                         double* d_alm_real_out, double* d_alm_imag_out);
void smoothalm_cuda_f32(int l_max, int n_fields, float fwhm_radians,
                         const float* d_alm_real_in, const float* d_alm_imag_in,
                         float* d_alm_real_out, float* d_alm_imag_out);

/**
 * In-place beam smoothing
 */
void smoothalm_inplace_cuda_f64(int l_max, int n_fields, double fwhm_radians,
                                 double* d_alm_real, double* d_alm_imag);
void smoothalm_inplace_cuda_f32(int l_max, int n_fields, float fwhm_radians,
                                 float* d_alm_real, float* d_alm_imag);

// ============================================================================
// ALM × f(l) Filter
// ============================================================================

/**
 * Multiply alm by arbitrary filter function f(l)
 *
 * @param l_max Maximum multipole
 * @param n_fields Number of fields
 * @param d_fl Filter values [l_max+1] (device)
 * @param d_log_fl Log of filter values (device, for log-space mode)
 * @param d_sign_fl Sign of filter values (device, for log-space mode)
 * @param d_alm_real_in Input real part (device)
 * @param d_alm_imag_in Input imag part (device)
 * @param d_alm_real_out Output real part (device)
 * @param d_alm_imag_out Output imag part (device)
 * @param use_log_space Use log-space computation
 */
void almxfl_cuda_f64(int l_max, int n_fields,
                      const double* d_fl, const double* d_log_fl, const int8_t* d_sign_fl,
                      const double* d_alm_real_in, const double* d_alm_imag_in,
                      double* d_alm_real_out, double* d_alm_imag_out,
                      bool use_log_space);
void almxfl_cuda_f32(int l_max, int n_fields,
                      const float* d_fl, const float* d_log_fl, const int8_t* d_sign_fl,
                      const float* d_alm_real_in, const float* d_alm_imag_in,
                      float* d_alm_real_out, float* d_alm_imag_out,
                      bool use_log_space);

/**
 * In-place almxfl (simple, no log-space)
 */
void almxfl_inplace_cuda_f64(int l_max, int n_fields, const double* d_fl,
                              double* d_alm_real, double* d_alm_imag);
void almxfl_inplace_cuda_f32(int l_max, int n_fields, const float* d_fl,
                              float* d_alm_real, float* d_alm_imag);

// ============================================================================
// Pixel Window Function
// ============================================================================

/**
 * Compute approximate pixel window function
 *
 * @param l_max Maximum multipole
 * @param nside HEALPix nside parameter
 * @param d_pixwin Output pixel window [l_max+1] (device)
 * @param d_log_pixwin Output log(pixel window) (device)
 */
void pixwin_cuda_f64(int l_max, int nside,
                      double* d_pixwin, double* d_log_pixwin);
void pixwin_cuda_f32(int l_max, int nside,
                      float* d_pixwin, float* d_log_pixwin);

// ============================================================================
// Resize ALM
// ============================================================================

/**
 * Resize alm to different lmax (truncate or zero-pad)
 *
 * @param l_max_in Input maximum multipole
 * @param l_max_out Output maximum multipole
 * @param n_fields Number of fields
 * @param d_alm_real_in Input real part [n_fields, l_max_in+1, l_max_in+1]
 * @param d_alm_imag_in Input imag part
 * @param d_alm_real_out Output real part [n_fields, l_max_out+1, l_max_out+1]
 * @param d_alm_imag_out Output imag part
 */
void resize_alm_cuda_f64(int l_max_in, int l_max_out, int n_fields,
                          const double* d_alm_real_in, const double* d_alm_imag_in,
                          double* d_alm_real_out, double* d_alm_imag_out);
void resize_alm_cuda_f32(int l_max_in, int l_max_out, int n_fields,
                          const float* d_alm_real_in, const float* d_alm_imag_in,
                          float* d_alm_real_out, float* d_alm_imag_out);

// ============================================================================
// Synalm: Generate Random ALM from Power Spectrum
// ============================================================================

/**
 * Initialize RNG states for synalm
 *
 * @param d_states Output pointer to device RNG states
 * @param n_states Number of states to allocate
 * @param seed Random seed
 */
void synalm_init_rng_cuda(curandState** d_states, int n_states, unsigned long long seed);

/**
 * Free RNG states
 */
void synalm_free_rng_cuda(curandState* d_states);

/**
 * Generate random alm from power spectrum
 *
 * @param l_max Maximum multipole
 * @param n_fields Number of fields
 * @param d_cl Power spectrum [n_fields, l_max+1] (device)
 * @param d_log_cl Log of power spectrum (device, for log-space mode)
 * @param rng_states RNG states (device)
 * @param d_alm_real Output real part [n_fields, l_max+1, l_max+1] (device)
 * @param d_alm_imag Output imag part (device)
 * @param use_log_space Use log-space computation for small Cl values
 */
void synalm_cuda_f64(int l_max, int n_fields,
                      const double* d_cl, const double* d_log_cl,
                      curandState* rng_states,
                      double* d_alm_real, double* d_alm_imag,
                      bool use_log_space);
void synalm_cuda_f32(int l_max, int n_fields,
                      const float* d_cl, const float* d_log_cl,
                      curandState* rng_states,
                      float* d_alm_real, float* d_alm_imag,
                      bool use_log_space);

/**
 * Generate correlated random alm using Cholesky decomposition
 *
 * For correlated fields (T, E, B with cross-spectra), the Cl matrix
 * is decomposed as C = L * L^T, and alm = L * g where g is independent Gaussian.
 *
 * @param l_max Maximum multipole
 * @param n_fields Number of correlated fields
 * @param d_cholesky Cholesky L matrix [l_max+1, n_fields, n_fields] (device)
 * @param rng_states RNG states (device)
 * @param d_alm_real Output real part [n_fields, l_max+1, l_max+1] (device)
 * @param d_alm_imag Output imag part (device)
 */
void synalm_correlated_cuda_f64(int l_max, int n_fields,
                                 const double* d_cholesky,
                                 curandState* rng_states,
                                 double* d_alm_real, double* d_alm_imag);
void synalm_correlated_cuda_f32(int l_max, int n_fields,
                                 const float* d_cholesky,
                                 curandState* rng_states,
                                 float* d_alm_real, float* d_alm_imag);

/**
 * Compute Cholesky decomposition of Cl covariance matrix (host-side)
 *
 * @param l_max Maximum multipole
 * @param n_fields Number of fields
 * @param cl_matrix Input covariance [l_max+1, n_fields, n_fields] (host)
 * @param cholesky Output Cholesky L matrix [l_max+1, n_fields, n_fields] (host)
 */
void compute_cholesky_f64(int l_max, int n_fields,
                           const double* cl_matrix, double* cholesky);
void compute_cholesky_f32(int l_max, int n_fields,
                           const float* cl_matrix, float* cholesky);

#ifdef __cplusplus
}
#endif

#endif // HEALPIX_OPS_H
