#ifndef FFT_GM_CUH
#define FFT_GM_CUH

#include <cufft.h>
#include "spht_types.h"

/**
 * Structure to hold cuFFT plans for different ring sizes
 */
typedef struct {
    int nside;
    int l_max;

    // Equatorial FFT plan (batched, all same size)
    cufftHandle equatorial_plan;
    int n_equatorial_rings;  // 2*nside + 1
    int equatorial_fft_size; // 4*nside

    // Polar FFT plans (one per unique size, or individual)
    // For simplicity, we can do polar rings individually
    cufftHandle* polar_plans;  // Array of nside-1 plans
    int n_polar_sizes;         // nside - 1

} fft_plans_t;

/**
 * Create FFT plans for a given nside
 */
fft_plans_t* create_fft_plans(int nside, int l_max);

/**
 * Destroy FFT plans
 */
void destroy_fft_plans(fft_plans_t* plans);

/**
 * Compute Gm for all equatorial rings using batched FFT
 *
 * Input:
 *   map_equatorial: [n_equatorial, 4*nside] real, device memory
 *   phi_0: [n_equatorial] real, first pixel azimuth for each ring
 *
 * Output:
 *   Gm_equatorial: [n_equatorial, l_max+1] complex, device memory
 *
 * Algorithm:
 *   1. Batched R2C FFT: fft_out[r, k] = FFT(map[r, :])[k]
 *   2. Phase correction: Gm[r, m] = fft_out[r, m] * exp(-i * m * phi_0[r])
 *   3. Truncate to l_max+1 coefficients (FFT gives 4*nside/2+1)
 */
void compute_gm_equatorial_fft(
    const fft_plans_t* plans,
    int n_maps,                    // Number of independent maps
    const real_t* map_equatorial,  // [n_maps, n_equatorial, 4*nside]
    const real_t* phi_0,           // [n_equatorial]
    complex_t* Gm_out              // [n_maps, n_equatorial, l_max+1]
);

/**
 * Compute Gm for polar rings (north-south pairs)
 *
 * Process north and south rings with same pixel count together.
 *
 * For ring pair (north_idx, south_idx) with N pixels:
 *   1. FFT both rings (batch of 2, size N)
 *   2. Apply phase correction
 *   3. Zero-pad Gm[m] for m >= N
 */
void compute_gm_polar_pairs_fft(
    int nside,
    int l_max,
    int n_maps,
    const real_t* map_in,          // [n_maps, n_rings, 4*nside] full map
    const ring_geometry_t* geom,
    complex_t* Gm_out              // [n_maps, n_rings, l_max+1]
);

/**
 * Apply phase correction after FFT with conjugate symmetry handling
 * Gm[r, m] = fft_out[r, m] * exp(-i * m * phi_0[r])
 *
 * For R2C FFT: output has N/2+1 coefficients, use conjugate symmetry for m > N/2
 */
__global__ void apply_phase_correction_kernel(
    int n_rings,
    int fft_input_size,  // Input size N (number of pixels)
    int fft_out_size,    // Output size from R2C FFT (N/2+1)
    int l_max,
    const cufftDoubleComplex* fft_out,  // [n_rings, fft_out_size]
    const real_t* phi_0,                 // [n_rings]
    complex_t* Gm_out                    // [n_rings, l_max+1]
);

/**
 * Compute Gm for all rings using FFT
 *
 * This is the main entry point that handles:
 * - Equatorial rings (batched FFT)
 * - Polar ring pairs (paired FFT with north-south batching)
 */
void compute_gm_all_rings(
    int nside,
    int l_max,
    int n_maps,
    const real_t* map_in,          // [n_maps, n_rings, 4*nside]
    const ring_geometry_t* geom,
    complex_t* Gm_out              // [n_maps, n_rings, l_max+1]
);

#endif // FFT_GM_CUH
