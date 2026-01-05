/**
 * alm2map: Spherical Harmonic Synthesis (alm to map transform)
 *
 * Reference: jax_healpix/SPHT_jax.py alm2ring_ns_dot() lines 305-310
 *
 * Einsum equivalent: "tlm,lmr,rjm->trj"
 * map[t,r,j] = sum_l sum_m (alm[t,l,m] * ylm[l,m,r] * phi[r,j,m])
 *
 * Decomposition:
 * 1. Fmy[t,m,r] = sum_l (alm[t,l,m] * ylm[l,m,r])
 * 2. map[t,r,j] = sum_m (Fmy[t,m,r] * phi[r,j,m])
 */

#include "../include/spht_types.h"
#include "../include/ylm_recurrence.cuh"
#include "../include/ring_processing.cuh"
#include <stdio.h>

// ============================================================================
// Step 1: Compute Fmy = sum_l (alm * ylm)
// ============================================================================

/**
 * Kernel for Fmy computation
 * Fmy[t,m,r] = sum_l (alm[t,l,m] * ylm[l,m,r])
 * Note: Only l >= m is valid
 */
__global__ void alm2ring_step1_fmy_kernel(int n_maps, int n_rings_batch, int l_max,
                                           const complex_t* alm,  // [n_maps, l_max+1, l_max+1]
                                           const real_t* ylm,     // [l_max+1, l_max+1, n_rings_batch]
                                           complex_t* Fmy) {      // [n_maps, l_max+1, n_rings_batch]
    int t = blockIdx.x;
    int m = blockIdx.y * blockDim.x + threadIdx.x;
    int r = blockIdx.z;

    if (t >= n_maps || m > l_max || r >= n_rings_batch) return;

    int lp1 = l_max + 1;

    complex_t sum;
    sum.x = 0.0;
    sum.y = 0.0;

    // Sum over l from m to l_max (only valid for l >= m)
    for (int l = m; l <= l_max; l++) {
        int alm_idx = INDEX_3D(t, l, m, lp1, lp1);
        int ylm_idx = INDEX_3D(l, m, r, lp1, n_rings_batch);

        real_t ylm_val = ylm[ylm_idx];
        sum.x += alm[alm_idx].x * ylm_val;
        sum.y += alm[alm_idx].y * ylm_val;
    }

    int fmy_idx = INDEX_3D(t, m, r, lp1, n_rings_batch);
    Fmy[fmy_idx] = sum;
}

// ============================================================================
// Step 2: Compute map = sum_m (Fmy * phi)
// ============================================================================

/**
 * Kernel for map synthesis
 * map[t,r,j] = Real(sum_m (Fmy[t,m,r] * phi[r,j,m]))
 */
__global__ void alm2ring_step2_synthesize_kernel(int n_maps, int n_rings_batch,
                                                   int nside, int l_max, int ring_start,
                                                   const complex_t* Fmy,   // [n_maps, l_max+1, n_rings_batch]
                                                   const complex_t* phi,   // [n_rings_batch, 4*nside, l_max+1]
                                                   const int* n_pixels,    // [n_rings]
                                                   real_t* map) {          // [n_maps, n_rings_batch, 4*nside]
    int t = blockIdx.x;
    int r = blockIdx.y;
    int j = blockIdx.z * blockDim.x + threadIdx.x;

    if (t >= n_maps || r >= n_rings_batch) return;

    int ring_idx = ring_start + r;
    int npix = n_pixels[ring_idx];

    if (j >= npix) return;

    int lp1 = l_max + 1;

    complex_t sum;
    sum.x = 0.0;
    sum.y = 0.0;

    for (int m = 0; m <= l_max; m++) {
        int fmy_idx = INDEX_3D(t, m, r, lp1, n_rings_batch);
        int phi_idx = INDEX_3D(r, j, m, 4 * nside, lp1);

        complex_t fmy_val = Fmy[fmy_idx];
        complex_t phi_val = phi[phi_idx];

        // Complex multiplication
        sum.x += fmy_val.x * phi_val.x - fmy_val.y * phi_val.y;
        sum.y += fmy_val.x * phi_val.y + fmy_val.y * phi_val.x;
    }

    // Only real part for spin-0 maps
    int map_idx = INDEX_3D(t, r, j, n_rings_batch, 4 * nside);
    map[map_idx] = sum.x;
}

/**
 * Kernel for complex map synthesis (spin-2)
 */
__global__ void alm2ring_step2_synthesize_complex_kernel(int n_maps, int n_rings_batch,
                                                           int nside, int l_max, int ring_start,
                                                           const complex_t* Fmy,
                                                           const complex_t* phi,
                                                           const int* n_pixels,
                                                           complex_t* map) {
    int t = blockIdx.x;
    int r = blockIdx.y;
    int j = blockIdx.z * blockDim.x + threadIdx.x;

    if (t >= n_maps || r >= n_rings_batch) return;

    int ring_idx = ring_start + r;
    int npix = n_pixels[ring_idx];

    if (j >= npix) return;

    int lp1 = l_max + 1;

    complex_t sum;
    sum.x = 0.0;
    sum.y = 0.0;

    for (int m = 0; m <= l_max; m++) {
        int fmy_idx = INDEX_3D(t, m, r, lp1, n_rings_batch);
        int phi_idx = INDEX_3D(r, j, m, 4 * nside, lp1);

        complex_t fmy_val = Fmy[fmy_idx];
        complex_t phi_val = phi[phi_idx];

        sum.x += fmy_val.x * phi_val.x - fmy_val.y * phi_val.y;
        sum.y += fmy_val.x * phi_val.y + fmy_val.y * phi_val.x;
    }

    int map_idx = INDEX_3D(t, r, j, n_rings_batch, 4 * nside);
    map[map_idx] = sum;
}

// ============================================================================
// Full alm2map Transform
// ============================================================================

/**
 * Perform alm2map for a batch of north rings
 */
void alm2ring_ns(int n_maps, int nside, int l_max,
                 int n_rings_batch, int ring_start,
                 const complex_t* alm,
                 const real_t* ylm,
                 const complex_t* phi,
                 const ring_geometry_t* geom,
                 real_t* map_out,
                 complex_t* Fmy_buffer) {
    int lp1 = l_max + 1;

    // Step 1: Compute Fmy
    dim3 block1(256);
    dim3 grid1(n_maps, CEILDIV(lp1, 256), n_rings_batch);

    alm2ring_step1_fmy_kernel<<<grid1, block1>>>(n_maps, n_rings_batch, l_max,
                                                   alm, ylm, Fmy_buffer);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Synthesize map
    dim3 block2(256);
    dim3 grid2(n_maps, n_rings_batch, CEILDIV(4 * nside, 256));

    alm2ring_step2_synthesize_kernel<<<grid2, block2>>>(n_maps, n_rings_batch, nside, l_max,
                                                          ring_start, Fmy_buffer, phi,
                                                          geom->n_pixels, map_out);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * Full alm2map transform
 *
 * @param nside HEALPix nside
 * @param l_max Maximum l
 * @param n_maps Number of maps
 * @param alm_in Input alm coefficients [n_maps, l_max+1, l_max+1]
 * @param map_out Output maps [n_maps, 4*nside-1, 4*nside]
 */
extern "C"
void alm2map_cuda(int nside, int l_max, int n_maps,
                   const complex_t* alm_in,
                   real_t* map_out) {
    int n_rings_total = 4 * nside - 1;
    int lp1 = l_max + 1;

    // Multiply alm by 2 for m > 0 to account for missing m < 0
    // (We only compute m >= 0 and use conjugate symmetry)
    complex_t* alm_scaled;
    size_t alm_size = (size_t)n_maps * lp1 * lp1 * sizeof(complex_t);
    CUDA_CHECK(cudaMalloc(&alm_scaled, alm_size));
    CUDA_CHECK(cudaMemcpy(alm_scaled, alm_in, alm_size, cudaMemcpyDeviceToDevice));

    // Scale m > 0 by 2 (done on host for simplicity)
    {
        complex_t* h_alm = new complex_t[n_maps * lp1 * lp1];
        CUDA_CHECK(cudaMemcpy(h_alm, alm_scaled, alm_size, cudaMemcpyDeviceToHost));
        for (int t = 0; t < n_maps; t++) {
            for (int l = 0; l <= l_max; l++) {
                for (int m = 1; m <= l; m++) {
                    int idx = INDEX_3D(t, l, m, lp1, lp1);
                    h_alm[idx].x *= 2.0;
                    h_alm[idx].y *= 2.0;
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(alm_scaled, h_alm, alm_size, cudaMemcpyHostToDevice));
        delete[] h_alm;
    }

    // Precompute ring geometry
    ring_geometry_t* geom = allocate_ring_geometry(nside);
    precompute_ring_geometry(nside, geom);

    // Allocate working buffers
    int n_batches = CEILDIV(2 * nside, RING_BATCH_SIZE);

    complex_t* phi_buffer;
    CUDA_CHECK(cudaMalloc(&phi_buffer, (size_t)RING_BATCH_SIZE * 4 * nside * lp1 * sizeof(complex_t)));

    complex_t* Fmy_buffer;
    CUDA_CHECK(cudaMalloc(&Fmy_buffer, (size_t)n_maps * lp1 * RING_BATCH_SIZE * sizeof(complex_t)));

    real_t* ylm_buffer;
    CUDA_CHECK(cudaMalloc(&ylm_buffer, (size_t)lp1 * lp1 * RING_BATCH_SIZE * sizeof(real_t)));

    real_t* ylm_south_buffer;
    CUDA_CHECK(cudaMalloc(&ylm_south_buffer, (size_t)lp1 * lp1 * RING_BATCH_SIZE * sizeof(real_t)));

    real_t* map_ring_buffer;
    CUDA_CHECK(cudaMalloc(&map_ring_buffer, (size_t)n_maps * RING_BATCH_SIZE * 4 * nside * sizeof(real_t)));

    // Initialize output map to zero
    CUDA_CHECK(cudaMemset(map_out, 0, (size_t)n_maps * n_rings_total * 4 * nside * sizeof(real_t)));

    // Process rings in batches
    for (int batch = 0; batch < n_batches; batch++) {
        int ring_start = batch * RING_BATCH_SIZE;
        int actual_rings = min(RING_BATCH_SIZE, 2 * nside - ring_start);

        if (actual_rings <= 0) break;

        // 1. Compute YLM for north rings
        compute_ylm_spin0(l_max, actual_rings,
                          geom->log_beta + ring_start,
                          geom->beta_sign + ring_start,
                          ylm_buffer);

        // 2. Compute phase factors (phase = +1 for alm2map)
        compute_phi_phases(nside, l_max, actual_rings, ring_start,
                           1, geom, phi_buffer);

        // 3. Process north rings
        alm2ring_ns(n_maps, nside, l_max, actual_rings, ring_start,
                    alm_scaled, ylm_buffer, phi_buffer, geom,
                    map_ring_buffer, Fmy_buffer);

        // Copy to output map (north rings)
        for (int r = 0; r < actual_rings; r++) {
            int ring_idx = ring_start + r;
            if (ring_idx >= 2 * nside) break;

            size_t src_offset = (size_t)r * 4 * nside;
            size_t dst_offset = (size_t)ring_idx * 4 * nside;

            for (int t = 0; t < n_maps; t++) {
                CUDA_CHECK(cudaMemcpy(
                    map_out + t * n_rings_total * 4 * nside + dst_offset,
                    map_ring_buffer + t * actual_rings * 4 * nside + src_offset,
                    4 * nside * sizeof(real_t),
                    cudaMemcpyDeviceToDevice
                ));
            }
        }

        // 4. Apply south symmetry to YLM (zeros out equator to prevent double-counting)
        apply_south_symmetry(l_max, actual_rings, nside, ring_start, ylm_buffer, ylm_south_buffer);

        // 5. Process south rings
        alm2ring_ns(n_maps, nside, l_max, actual_rings, ring_start,
                    alm_scaled, ylm_south_buffer, phi_buffer, geom,
                    map_ring_buffer, Fmy_buffer);

        // Copy to output map (south rings)
        // South ring index = 4*nside - 1 - north_ring_index
        // But we need to handle equator specially (don't duplicate)
        for (int r = 0; r < actual_rings; r++) {
            int north_ring_idx = ring_start + r;
            if (north_ring_idx >= 2 * nside) break;

            // Skip equator (ring_i = 2*nside, which is north_ring_idx = 2*nside-1)
            if (north_ring_idx == 2 * nside - 1) continue;

            int south_ring_idx = 4 * nside - 2 - north_ring_idx;  // 0-indexed

            size_t src_offset = (size_t)r * 4 * nside;
            size_t dst_offset = (size_t)south_ring_idx * 4 * nside;

            for (int t = 0; t < n_maps; t++) {
                CUDA_CHECK(cudaMemcpy(
                    map_out + t * n_rings_total * 4 * nside + dst_offset,
                    map_ring_buffer + t * actual_rings * 4 * nside + src_offset,
                    4 * nside * sizeof(real_t),
                    cudaMemcpyDeviceToDevice
                ));
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(alm_scaled));
    CUDA_CHECK(cudaFree(phi_buffer));
    CUDA_CHECK(cudaFree(Fmy_buffer));
    CUDA_CHECK(cudaFree(ylm_buffer));
    CUDA_CHECK(cudaFree(ylm_south_buffer));
    CUDA_CHECK(cudaFree(map_ring_buffer));
    free_ring_geometry(geom);
}
