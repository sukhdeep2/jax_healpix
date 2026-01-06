/**
 * Unified Ring Geometry Computation for HEALPix
 *
 * Consolidates ring geometry computation that was previously duplicated
 * in map2alm_v6.cu, alm2map_v6.cu, bluestein_fft.cu, and fft_gm.cu.
 *
 * HEALPix ring structure for nside:
 *   - Total rings: 4*nside - 1
 *   - North polar cap: rings 0 to nside-2 (nside-1 rings)
 *   - Equatorial belt: rings nside-1 to 3*nside-1 (2*nside+1 rings)
 *   - South polar cap: rings 3*nside to 4*nside-2 (nside-1 rings)
 */

#ifndef RING_GEOMETRY_CUH
#define RING_GEOMETRY_CUH

#include <cuda_runtime.h>
#include "precision_traits.cuh"

// ============================================================================
// Core ring geometry computation
// ============================================================================

/**
 * Compute ring geometry for a given ring index.
 *
 * @param ring_idx   0-indexed ring number (0 to 4*nside-2)
 * @param nside      HEALPix resolution parameter
 * @param cos_theta  Output: cos(theta) for this ring
 * @param sin_theta  Output: sin(theta) for this ring
 * @param phi_0      Output: starting azimuthal angle phi_0
 * @param n_pixels   Output: number of pixels in this ring
 *
 * @tparam T         Precision type (double, float)
 */
template<typename T>
__device__ __forceinline__ void compute_ring_geometry(
    int ring_idx, int nside,
    T* cos_theta, T* sin_theta, T* phi_0, int* n_pixels
) {
    using Traits = PrecisionTraits<T>;
    using C = typename Traits::compute_t;

    int ring_i = ring_idx + 1;  // Convert to 1-indexed (HEALPix convention)
    C cos_th, sin_th, phi0;
    int npix;

    if (ring_i < nside) {
        // North polar cap: rings 1 to nside-1 (0-indexed: 0 to nside-2)
        // cos(theta) = 1 - i²/(3*nside²)
        // phi_0 = pi/(4*i)
        // n_pixels = 4*i
        C i2_3n2 = C(ring_i * ring_i) / C(3.0 * nside * nside);
        cos_th = C(1.0) - i2_3n2;
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        phi0 = C(Traits::PI_VAL) / C(2.0 * ring_i) * C(0.5);  // = pi/(4*i)
        npix = 4 * ring_i;
    } else if (ring_i > 3 * nside) {
        // South polar cap: rings 3*nside+1 to 4*nside-1 (0-indexed: 3*nside to 4*nside-2)
        // Mirror of north polar cap
        int mirror_i = 4 * nside - ring_i;
        C i2_3n2 = C(mirror_i * mirror_i) / C(3.0 * nside * nside);
        cos_th = -(C(1.0) - i2_3n2);  // Negative cos_theta for southern hemisphere
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        phi0 = C(Traits::PI_VAL) / C(2.0 * mirror_i) * C(0.5);
        npix = 4 * mirror_i;
    } else {
        // Equatorial belt: rings nside to 3*nside (0-indexed: nside-1 to 3*nside-1)
        // cos(theta) = 4/3 - 2*i/(3*nside)
        // phi_0 = pi/(4*nside) * (1 - s/2) where s = 1 if even ring, 2 if odd
        // n_pixels = 4*nside
        cos_th = C(4.0 / 3.0) - C(2.0 * ring_i) / C(3.0 * nside);
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        int s = (ring_i % 2 == 0) ? 1 : 2;
        phi0 = C(Traits::PI_VAL) / C(2.0 * nside) * C(1.0 - s / 2.0);
        npix = 4 * nside;
    }

    *cos_theta = T(cos_th);
    *sin_theta = T(sin_th);
    *phi_0 = T(phi0);
    *n_pixels = npix;
}

/**
 * Simplified version returning only cos_theta and sin_theta.
 */
template<typename T>
__device__ __forceinline__ void compute_ring_angles(
    int ring_idx, int nside,
    T* cos_theta, T* sin_theta
) {
    T phi0_unused;
    int npix_unused;
    compute_ring_geometry<T>(ring_idx, nside, cos_theta, sin_theta, &phi0_unused, &npix_unused);
}

/**
 * Get the number of pixels in a ring.
 */
__device__ __forceinline__ int get_ring_size(int ring_idx, int nside) {
    int ring_i = ring_idx + 1;
    if (ring_i < nside) {
        return 4 * ring_i;
    } else if (ring_i > 3 * nside) {
        return 4 * (4 * nside - ring_i);
    } else {
        return 4 * nside;
    }
}

/**
 * Get the phi_0 (starting azimuthal angle) for a ring.
 */
template<typename T>
__device__ __forceinline__ T get_ring_phi0(int ring_idx, int nside) {
    using Traits = PrecisionTraits<T>;
    using C = typename Traits::compute_t;

    int ring_i = ring_idx + 1;
    C phi0;

    if (ring_i < nside) {
        phi0 = C(Traits::PI_VAL) / C(2.0 * ring_i) * C(0.5);
    } else if (ring_i > 3 * nside) {
        int mirror_i = 4 * nside - ring_i;
        phi0 = C(Traits::PI_VAL) / C(2.0 * mirror_i) * C(0.5);
    } else {
        int s = (ring_i % 2 == 0) ? 1 : 2;
        phi0 = C(Traits::PI_VAL) / C(2.0 * nside) * C(1.0 - s / 2.0);
    }

    return T(phi0);
}

// ============================================================================
// Ring indexing helpers
// ============================================================================

/**
 * Get the corresponding north ring index for symmetry operations.
 * For a ring in the southern hemisphere, returns its northern mirror.
 */
__device__ __forceinline__ int get_north_ring_index(int ring_idx, int n_rings) {
    int n_north_rings = (n_rings + 1) / 2;
    if (ring_idx < n_north_rings) {
        return ring_idx;
    } else {
        return n_rings - 1 - ring_idx;
    }
}

/**
 * Check if a ring is on the equator (unique ring with no southern pair).
 */
__device__ __forceinline__ bool is_equator_ring(int ring_idx, int nside) {
    return ring_idx == 2 * nside - 1;
}

/**
 * Get the number of north rings (including equator).
 */
__host__ __device__ __forceinline__ int get_n_north_rings(int nside) {
    return 2 * nside;
}

/**
 * Get the total number of rings.
 */
__host__ __device__ __forceinline__ int get_n_rings(int nside) {
    return 4 * nside - 1;
}

/**
 * Get the maximum pixels in any ring (equatorial ring size).
 */
__host__ __device__ __forceinline__ int get_max_ring_size(int nside) {
    return 4 * nside;
}

// ============================================================================
// Ring geometry with log-arithmetic (for numerical stability at high l)
// ============================================================================

/**
 * Compute ring geometry using log-arithmetic for better numerical stability.
 * Useful for synthesis kernels with large l_max.
 */
template<typename T>
__device__ __forceinline__ void compute_ring_geometry_log(
    int ring_idx, int nside,
    T* cos_theta, T* sin_theta, T* phi_0, int* n_pixels
) {
    using Traits = PrecisionTraits<T>;
    using C = typename Traits::compute_t;

    int ring_i = ring_idx + 1;
    C cos_th, sin_th, phi0;
    int npix;

    C log_nside = Traits::log_d(C(nside));

    if (ring_i < nside) {
        // North polar cap with log-arithmetic
        C log_ring_i = Traits::log_d(C(ring_i));
        C i2_3n2 = Traits::exp_d(C(2.0) * log_ring_i - Traits::LOG_3_VAL - C(2.0) * log_nside);
        cos_th = C(1.0) - i2_3n2;
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        // phi0 = PI / (4 * ring_i) = exp(log(PI) - log(4) - log(ring_i))
        phi0 = Traits::exp_d(Traits::LOG_PI_VAL - Traits::LOG_4_VAL - log_ring_i);
        npix = 4 * ring_i;
    } else if (ring_i > 3 * nside) {
        // South polar cap with log-arithmetic
        int mirror_i = 4 * nside - ring_i;
        C log_mirror_i = Traits::log_d(C(mirror_i));
        C i2_3n2 = Traits::exp_d(C(2.0) * log_mirror_i - Traits::LOG_3_VAL - C(2.0) * log_nside);
        cos_th = -(C(1.0) - i2_3n2);
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        phi0 = Traits::exp_d(Traits::LOG_PI_VAL - Traits::LOG_4_VAL - log_mirror_i);
        npix = 4 * mirror_i;
    } else {
        // Equatorial belt with log-arithmetic
        C log_ring_i = Traits::log_d(C(ring_i));
        C term = Traits::exp_d(Traits::LOG_2_VAL + log_ring_i - Traits::LOG_3_VAL - log_nside);
        cos_th = C(4.0 / 3.0) - term;
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        int s = (ring_i % 2 == 0) ? 1 : 2;
        phi0 = Traits::exp_d(Traits::LOG_PI_VAL - Traits::LOG_2_VAL - log_nside) * C(1.0 - s / 2.0);
        npix = 4 * nside;
    }

    *cos_theta = T(cos_th);
    *sin_theta = T(sin_th);
    *phi_0 = T(phi0);
    *n_pixels = npix;
}

#endif // RING_GEOMETRY_CUH
