/**
 * Ring Processing Kernels for SPHT
 *
 * Implements:
 * - Phase factor computation
 * - North/South ring symmetry
 * - Ring-based transform operations
 */

#include "../include/ring_processing.cuh"
#include "../include/spht_types.h"
#include <stdio.h>

// ============================================================================
// Helper Functions
// ============================================================================

__device__ __host__ int get_n_pixels(int ring_i, int nside) {
    // ring_i is 1-indexed
    if (ring_i < nside) {
        return 4 * ring_i;
    } else if (ring_i > 3 * nside) {
        return 4 * (4 * nside - ring_i);
    } else {
        return 4 * nside;
    }
}

__device__ __host__ int get_ring_start_pixel(int ring_i, int nside) {
    // ring_i is 1-indexed
    if (ring_i < nside) {
        // North polar cap: sum of 4*k for k=1 to ring_i-1
        return 2 * ring_i * (ring_i - 1);
    } else if (ring_i <= 3 * nside) {
        // Equatorial belt
        int north_cap_pixels = 2 * nside * (nside - 1);
        return north_cap_pixels + (ring_i - nside) * 4 * nside;
    } else {
        // South polar cap
        int total_pixels = 12 * nside * nside;
        int remaining_rings = 4 * nside - ring_i;
        int south_cap_remaining = 2 * remaining_rings * (remaining_rings + 1);
        return total_pixels - south_cap_remaining;
    }
}

// ============================================================================
// Phase Factor Computation
// ============================================================================

/**
 * Kernel to compute phase factors
 * phi_out[ring, j, m] = exp(i * phase * m * phi_j)
 * where phi_j = phi_0 + 2*pi*j / n_pixels
 */
__global__ void compute_phi_phases_kernel(int nside, int l_max, int n_rings_batch,
                                           int ring_start, int phase,
                                           const real_t* phi_0,
                                           const int* n_pixels,
                                           complex_t* phi_out) {
    int local_ring = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;  // Pixel index
    int m = blockIdx.z * blockDim.y + threadIdx.y;  // m value

    if (local_ring >= n_rings_batch) return;

    int ring_idx = ring_start + local_ring;  // 0-indexed into geometry arrays
    int ring_i = ring_idx + 1;               // 1-indexed ring number
    int npix = n_pixels[ring_idx];

    if (j >= npix || m > l_max) return;

    // phi_j = phi_0 + 2*pi*j / n_pixels
    real_t phi_j = phi_0[ring_idx] + 2.0 * PI * j / npix;

    // exp(i * phase * m * phi_j)
    real_t angle = phase * m * phi_j;
    int idx = INDEX_3D(local_ring, j, m, 4 * nside, l_max + 1);

    phi_out[idx].x = cos(angle);  // Real part
    phi_out[idx].y = sin(angle);  // Imag part
}

/**
 * Kernel to zero-pad phi values for pixels beyond ring's n_pixels
 */
__global__ void zero_pad_phi_kernel(int nside, int l_max, int n_rings_batch,
                                     int ring_start, const int* n_pixels,
                                     complex_t* phi_out) {
    int local_ring = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    int m = blockIdx.z * blockDim.y + threadIdx.y;

    if (local_ring >= n_rings_batch || j >= 4 * nside || m > l_max) return;

    int ring_idx = ring_start + local_ring;
    int npix = n_pixels[ring_idx];

    if (j >= npix) {
        int idx = INDEX_3D(local_ring, j, m, 4 * nside, l_max + 1);
        phi_out[idx].x = 0.0;
        phi_out[idx].y = 0.0;
    }
}

void compute_phi_phases(int nside, int l_max, int n_rings_batch, int ring_start,
                        int phase, const ring_geometry_t* geom,
                        complex_t* phi_out) {
    // Block and grid dimensions
    dim3 block(32, 8);  // 256 threads
    dim3 grid(n_rings_batch,
              CEILDIV(4 * nside, 32),
              CEILDIV(l_max + 1, 8));

    compute_phi_phases_kernel<<<grid, block>>>(nside, l_max, n_rings_batch, ring_start,
                                                phase, geom->phi_0, geom->n_pixels,
                                                phi_out);
    CUDA_CHECK(cudaGetLastError());

    // Zero pad beyond valid pixels
    zero_pad_phi_kernel<<<grid, block>>>(nside, l_max, n_rings_batch, ring_start,
                                          geom->n_pixels, phi_out);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// South Ring Symmetry
// ============================================================================

/**
 * Kernel to apply south symmetry
 * Y_{l,m}(-cos(theta)) = (-1)^{l+m} * Y_{l,m}(cos(theta))
 *
 * CRITICAL: Zero out equator ring to prevent double-counting
 * Reference: jax_healpix/SPHT_jax.py south_ring_ylm() line 154-156
 */
__global__ void apply_south_symmetry_kernel(int l_max, int n_rings_batch,
                                             int nside, int ring_start,
                                             const real_t* ylm_north,
                                             real_t* ylm_south) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int local_ring = blockIdx.z;

    if (l > l_max || m > l || local_ring >= n_rings_batch) return;

    int lp1 = l_max + 1;
    int idx = INDEX_3D(l, m, local_ring, lp1, n_rings_batch);

    // Get the north ring index (0-indexed)
    int north_ring_idx = ring_start + local_ring;

    // Zero out equator (ring_i = 2*nside in 1-indexed, or 2*nside-1 in 0-indexed)
    // to prevent double-counting. JAX: "no repeat on ring at equator"
    if (north_ring_idx >= 2 * nside - 1) {
        ylm_south[idx] = 0.0;
        return;
    }

    // Apply (-1)^{l+m} factor
    real_t sign = ((l + m) % 2 == 0) ? 1.0 : -1.0;
    ylm_south[idx] = sign * ylm_north[idx];
}

void apply_south_symmetry(int l_max, int n_rings_batch, int nside, int ring_start,
                          const real_t* ylm_north,
                          real_t* ylm_south) {
    dim3 block(16, 16, 1);
    dim3 grid(CEILDIV(l_max + 1, 16), CEILDIV(l_max + 1, 16), n_rings_batch);

    apply_south_symmetry_kernel<<<grid, block>>>(l_max, n_rings_batch, nside, ring_start,
                                                   ylm_north, ylm_south);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * Kernel for spin-2 south symmetry
 * _2Y has same symmetry as spin-0
 * _(-2)Y has additional -1 factor
 */
__global__ void apply_south_symmetry_spin2_kernel(int l_max, int n_rings_batch,
                                                    const real_t* ylm_p2_north,
                                                    const real_t* ylm_m2_north,
                                                    real_t* ylm_p2_south,
                                                    real_t* ylm_m2_south) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int ring = blockIdx.z;

    if (l > l_max || m > l || ring >= n_rings_batch) return;

    int lp1 = l_max + 1;
    int idx = INDEX_3D(l, m, ring, lp1, n_rings_batch);

    // Base symmetry factor
    real_t sign = ((l + m) % 2 == 0) ? 1.0 : -1.0;

    // +2 spin: same as spin-0
    ylm_p2_south[idx] = sign * ylm_p2_north[idx];

    // -2 spin: additional -1 factor (see SPHT_jax.py line 158)
    ylm_m2_south[idx] = -1.0 * sign * ylm_m2_north[idx];
}

void apply_south_symmetry_spin2(int l_max, int n_rings_batch,
                                 const real_t* ylm_p2_north,
                                 const real_t* ylm_m2_north,
                                 real_t* ylm_p2_south,
                                 real_t* ylm_m2_south) {
    dim3 block(16, 16, 1);
    dim3 grid(CEILDIV(l_max + 1, 16), CEILDIV(l_max + 1, 16), n_rings_batch);

    apply_south_symmetry_spin2_kernel<<<grid, block>>>(l_max, n_rings_batch,
                                                         ylm_p2_north, ylm_m2_north,
                                                         ylm_p2_south, ylm_m2_south);
    CUDA_CHECK(cudaGetLastError());
}
