#ifndef RING_PROCESSING_CUH
#define RING_PROCESSING_CUH

#include "spht_types.h"

/**
 * Compute phase factors exp(i * phase * m * phi_j) for all pixels in rings
 * Reference: jax_healpix/SPHT_jax.py phi_m() lines 104-125
 *
 * @param nside HEALPix nside parameter
 * @param l_max Maximum l value
 * @param n_rings_batch Number of rings in this batch
 * @param ring_start Starting ring index (0-indexed)
 * @param phase Phase factor (+1 for alm2map, -1 for map2alm)
 * @param geom Ring geometry structure
 * @param phi_out Output array [n_rings_batch, 4*nside, l_max+1] (complex)
 */
void compute_phi_phases(int nside, int l_max, int n_rings_batch, int ring_start,
                        int phase, const ring_geometry_t* geom,
                        complex_t* phi_out);

/**
 * Apply south symmetry to YLM values
 * Reference: jax_healpix/SPHT_jax.py south_ring_ylm() lines 145-161
 *
 * Y_{l,m}(-cos(theta)) = (-1)^{l+m} * Y_{l,m}(cos(theta))
 *
 * CRITICAL: Zeros out equator ring to prevent double-counting
 *
 * @param l_max Maximum l value
 * @param n_rings_batch Number of rings in batch
 * @param nside HEALPix nside parameter
 * @param ring_start Starting ring index (0-indexed)
 * @param ylm_north Input YLM for north rings [l_max+1, l_max+1, n_rings_batch]
 * @param ylm_south Output YLM for south rings [l_max+1, l_max+1, n_rings_batch]
 */
void apply_south_symmetry(int l_max, int n_rings_batch, int nside, int ring_start,
                          const real_t* ylm_north,
                          real_t* ylm_south);

/**
 * Apply south symmetry for spin-2 YLM
 * Additional -1 factor for _(-2)Y
 */
void apply_south_symmetry_spin2(int l_max, int n_rings_batch,
                                 const real_t* ylm_p2_north,
                                 const real_t* ylm_m2_north,
                                 real_t* ylm_p2_south,
                                 real_t* ylm_m2_south);

/**
 * Get number of pixels in a ring
 */
__device__ __host__ int get_n_pixels(int ring_i, int nside);

/**
 * Get starting pixel index for a ring
 */
__device__ __host__ int get_ring_start_pixel(int ring_i, int nside);

#endif // RING_PROCESSING_CUH
