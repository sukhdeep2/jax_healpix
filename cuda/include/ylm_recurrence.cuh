#ifndef YLM_RECURRENCE_CUH
#define YLM_RECURRENCE_CUH

#include "spht_types.h"
#include "log_arithmetic.cuh"

/**
 * Allocate ring geometry structure on device
 */
ring_geometry_t* allocate_ring_geometry(int nside);

/**
 * Free ring geometry structure
 */
void free_ring_geometry(ring_geometry_t* geom);

/**
 * Precompute ring geometry for given nside
 * Reference: jax_healpix/SPHT_jax.py ring_log_beta()
 */
void precompute_ring_geometry(int nside, ring_geometry_t* geom);

/**
 * Allocate YLM log structure on device
 */
ylm_log_t* allocate_ylm_log(int l_max, int n_rings);

/**
 * Free YLM log structure
 */
void free_ylm_log(ylm_log_t* ylm);

/**
 * Compute diagonal elements Y_{l,l} for all l from 1 to l_max
 * Reference: jax_healpix/YLM_jax_log.py sYLM_ll0_log() lines 29-62
 * Equation 15 from arXiv:1010.2084
 */
void sYLM_ll0_log(int l_max, int n_rings,
                  const real_t* log_beta_s,
                  const real_t* log_beta,
                  const int8_t* beta_sign,
                  ylm_log_t* ylm);

/**
 * Compute off-diagonal elements using recurrence
 * Reference: jax_healpix/YLM_jax_log.py sYLM_l0_log() lines 76-100
 * Equation 13-14 from arXiv:1010.2084
 */
void sYLM_l0_all(int l_max, int n_rings,
                 const real_t* log_beta,
                 const int8_t* beta_sign,
                 ylm_log_t* ylm);

/**
 * Compute spin-2 harmonics from spin-0
 * Reference: jax_healpix/YLM_jax_log.py sYLM_l2() lines 112-173
 * Equation A7 from arXiv:astro-ph/0502469
 */
void sYLM_l2_compute(int l_max, int n_rings,
                     const real_t* log_beta,
                     const real_t* log_beta_s2,  // 2 * log(sin(theta))
                     const int8_t* beta_sign,
                     const ylm_log_t* ylm_spin0,
                     real_t* ylm_spin2_p,   // +2 spin output
                     int8_t* sign_spin2_p,
                     real_t* ylm_spin2_m,   // -2 spin output
                     int8_t* sign_spin2_m);

/**
 * Normalize YLM values and convert from log to linear space
 * Applies 1/sqrt(4*pi) normalization
 */
void ylm_normalize(int l_max, int n_rings,
                   const ylm_log_t* ylm_log,
                   real_t* ylm_out);

/**
 * Complete YLM computation for spin-0
 * Combines all steps: diagonal, off-diagonal, normalization
 */
void compute_ylm_spin0(int l_max, int n_rings,
                       const real_t* log_beta,
                       const int8_t* beta_sign,
                       real_t* ylm_out);

/**
 * Complete YLM computation for spin-0 and spin-2
 */
void compute_ylm_all_spins(int l_max, int n_rings,
                           const real_t* log_beta,
                           const int8_t* beta_sign,
                           real_t* ylm_spin0_out,
                           real_t* ylm_spin2_p_out,
                           real_t* ylm_spin2_m_out);

#endif // YLM_RECURRENCE_CUH
