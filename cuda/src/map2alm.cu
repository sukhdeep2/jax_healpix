/**
 * map2alm: Spherical Harmonic Analysis (map to alm transform)
 *
 * Reference: jax_healpix/SPHT_jax.py ring2alm_ns_dot() lines 178-190
 *
 * Einsum equivalent: "trj,rjm,lmr->tlm"
 * alm[t,l,m] = sum_r sum_j (map[t,r,j] * phi[r,j,m] * ylm[l,m,r])
 *
 * Decomposition:
 * 1. Gmy[t,r,m] = sum_j (map[t,r,j] * phi[r,j,m])
 * 2. alm[t,l,m] += sum_r (Gmy[t,r,m] * ylm[l,m,r])
 */

#include "../include/spht_types.h"
#include "../include/ylm_recurrence.cuh"
#include "../include/ring_processing.cuh"
#include <stdio.h>

// ============================================================================
// Step 1: Compute Gmy = sum_j (map * phi)
// ============================================================================

/**
 * Kernel for Gmy computation
 * Gmy[t,r,m] = sum_j (map[t,r,j] * phi[r,j,m])
 */
__global__ void ring2alm_step1_gmy_kernel(int n_maps, int n_rings_batch,
                                           int nside, int l_max, int ring_start,
                                           const real_t* map,     // [n_maps, n_rings_batch, 4*nside]
                                           const complex_t* phi,  // [n_rings_batch, 4*nside, l_max+1]
                                           const int* n_pixels,   // [n_rings]
                                           complex_t* Gmy) {      // [n_maps, n_rings_batch, l_max+1]
    int t = blockIdx.x;
    int r = blockIdx.y;
    int m = blockIdx.z * blockDim.x + threadIdx.x;

    if (t >= n_maps || r >= n_rings_batch || m > l_max) return;

    int ring_idx = ring_start + r;
    int npix = n_pixels[ring_idx];
    int lp1 = l_max + 1;

    complex_t sum;
    sum.x = 0.0;
    sum.y = 0.0;

    for (int j = 0; j < npix; j++) {
        int map_idx = INDEX_3D(t, r, j, n_rings_batch, 4 * nside);
        int phi_idx = INDEX_3D(r, j, m, 4 * nside, lp1);

        real_t map_val = map[map_idx];
        complex_t phi_val = phi[phi_idx];

        sum.x += map_val * phi_val.x;
        sum.y += map_val * phi_val.y;
    }

    int gmy_idx = INDEX_3D(t, r, m, n_rings_batch, lp1);
    Gmy[gmy_idx] = sum;
}

// ============================================================================
// Step 2: Accumulate alm = sum_r (Gmy * ylm)
// ============================================================================

/**
 * Kernel to accumulate alm contributions from a ring batch
 * alm[t,l,m] += sum_r (Gmy[t,r,m] * ylm[l,m,r])
 */
__global__ void ring2alm_step2_accumulate_kernel(int n_maps, int n_rings_batch, int l_max,
                                                   const complex_t* Gmy,  // [n_maps, n_rings_batch, l_max+1]
                                                   const real_t* ylm,     // [l_max+1, l_max+1, n_rings_batch]
                                                   complex_t* alm) {      // [n_maps, l_max+1, l_max+1]
    int t = blockIdx.x;
    int l = blockIdx.y * blockDim.x + threadIdx.x;
    int m = blockIdx.z * blockDim.y + threadIdx.y;

    if (t >= n_maps || l > l_max || m > l) return;

    int lp1 = l_max + 1;

    complex_t sum;
    sum.x = 0.0;
    sum.y = 0.0;

    for (int r = 0; r < n_rings_batch; r++) {
        int ylm_idx = INDEX_3D(l, m, r, lp1, n_rings_batch);
        int gmy_idx = INDEX_3D(t, r, m, n_rings_batch, lp1);

        real_t ylm_val = ylm[ylm_idx];
        complex_t gmy_val = Gmy[gmy_idx];

        sum.x += gmy_val.x * ylm_val;
        sum.y += gmy_val.y * ylm_val;
    }

    // Atomic add to alm (multiple ring batches contribute)
    int alm_idx = INDEX_3D(t, l, m, lp1, lp1);
    atomicAdd(&alm[alm_idx].x, sum.x);
    atomicAdd(&alm[alm_idx].y, sum.y);
}

// ============================================================================
// Full map2alm Transform
// ============================================================================

/**
 * Perform map2alm for a batch of rings (north or south)
 */
void ring2alm_ns(int n_maps, int nside, int l_max,
                 int n_rings_batch, int ring_start,
                 const real_t* map_batch,
                 const real_t* ylm,
                 const complex_t* phi,
                 const ring_geometry_t* geom,
                 complex_t* alm_out,
                 complex_t* Gmy_buffer) {
    int lp1 = l_max + 1;

    // Step 1: Compute Gmy
    dim3 block1(256);
    dim3 grid1(n_maps, n_rings_batch, CEILDIV(lp1, 256));

    ring2alm_step1_gmy_kernel<<<grid1, block1>>>(n_maps, n_rings_batch, nside, l_max,
                                                   ring_start, map_batch, phi,
                                                   geom->n_pixels, Gmy_buffer);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Accumulate to alm
    dim3 block2(16, 16);
    dim3 grid2(n_maps, CEILDIV(lp1, 16), CEILDIV(lp1, 16));

    ring2alm_step2_accumulate_kernel<<<grid2, block2>>>(n_maps, n_rings_batch, l_max,
                                                          Gmy_buffer, ylm, alm_out);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * Full map2alm transform
 *
 * @param nside HEALPix nside
 * @param l_max Maximum l
 * @param n_maps Number of maps
 * @param map_in Input maps [n_maps, 4*nside-1, 4*nside]
 * @param alm_out Output alm coefficients [n_maps, l_max+1, l_max+1]
 */
extern "C"
void map2alm_cuda(int nside, int l_max, int n_maps,
                   const real_t* map_in,
                   complex_t* alm_out) {
    int n_rings_total = 4 * nside - 1;
    int lp1 = l_max + 1;

    // Initialize alm to zero
    size_t alm_size = (size_t)n_maps * lp1 * lp1 * sizeof(complex_t);
    CUDA_CHECK(cudaMemset(alm_out, 0, alm_size));

    // Precompute ring geometry
    ring_geometry_t* geom = allocate_ring_geometry(nside);
    precompute_ring_geometry(nside, geom);

    // Allocate working buffers
    int n_batches = CEILDIV(2 * nside, RING_BATCH_SIZE);

    complex_t* phi_buffer;
    CUDA_CHECK(cudaMalloc(&phi_buffer, (size_t)RING_BATCH_SIZE * 4 * nside * lp1 * sizeof(complex_t)));

    complex_t* Gmy_buffer;
    CUDA_CHECK(cudaMalloc(&Gmy_buffer, (size_t)n_maps * RING_BATCH_SIZE * lp1 * sizeof(complex_t)));

    real_t* ylm_buffer;
    CUDA_CHECK(cudaMalloc(&ylm_buffer, (size_t)lp1 * lp1 * RING_BATCH_SIZE * sizeof(real_t)));

    real_t* ylm_south_buffer;
    CUDA_CHECK(cudaMalloc(&ylm_south_buffer, (size_t)lp1 * lp1 * RING_BATCH_SIZE * sizeof(real_t)));

    real_t* map_ring_buffer;
    CUDA_CHECK(cudaMalloc(&map_ring_buffer, (size_t)n_maps * RING_BATCH_SIZE * 4 * nside * sizeof(real_t)));

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

        // 2. Compute phase factors (phase = -1 for map2alm)
        compute_phi_phases(nside, l_max, actual_rings, ring_start,
                           -1, geom, phi_buffer);

        // 3. Copy north ring map data to buffer
        for (int r = 0; r < actual_rings; r++) {
            int ring_idx = ring_start + r;
            if (ring_idx >= 2 * nside) break;

            size_t src_offset = (size_t)ring_idx * 4 * nside;
            size_t dst_offset = (size_t)r * 4 * nside;

            for (int t = 0; t < n_maps; t++) {
                CUDA_CHECK(cudaMemcpy(
                    map_ring_buffer + t * actual_rings * 4 * nside + dst_offset,
                    map_in + t * n_rings_total * 4 * nside + src_offset,
                    4 * nside * sizeof(real_t),
                    cudaMemcpyDeviceToDevice
                ));
            }
        }

        // 4. Process north rings
        ring2alm_ns(n_maps, nside, l_max, actual_rings, ring_start,
                    map_ring_buffer, ylm_buffer, phi_buffer, geom,
                    alm_out, Gmy_buffer);

        // 5. Apply south symmetry to YLM (zeros out equator to prevent double-counting)
        apply_south_symmetry(l_max, actual_rings, nside, ring_start, ylm_buffer, ylm_south_buffer);

        // 6. Copy south ring map data to buffer
        // South ring index = 4*nside - 2 - north_ring_index (0-indexed)
        for (int r = 0; r < actual_rings; r++) {
            int north_ring_idx = ring_start + r;
            if (north_ring_idx >= 2 * nside) break;

            // Skip equator (don't process twice)
            int south_ring_idx = 4 * nside - 2 - north_ring_idx;

            // If south ring is same as north (equator), zero out the buffer
            // Actually, for map2alm, the south contribution should still be computed
            // but the YLM south symmetry zeros out the equator contribution
            if (north_ring_idx == 2 * nside - 1) {
                // Equator: set ylm_south to 0 (already handled in south_ring_ylm)
                continue;
            }

            size_t src_offset = (size_t)south_ring_idx * 4 * nside;
            size_t dst_offset = (size_t)r * 4 * nside;

            for (int t = 0; t < n_maps; t++) {
                CUDA_CHECK(cudaMemcpy(
                    map_ring_buffer + t * actual_rings * 4 * nside + dst_offset,
                    map_in + t * n_rings_total * 4 * nside + src_offset,
                    4 * nside * sizeof(real_t),
                    cudaMemcpyDeviceToDevice
                ));
            }
        }

        // 7. Process south rings
        ring2alm_ns(n_maps, nside, l_max, actual_rings, ring_start,
                    map_ring_buffer, ylm_south_buffer, phi_buffer, geom,
                    alm_out, Gmy_buffer);
    }

    // Apply pixel area normalization
    // pix_area = 4 * PI / (12 * nside * nside)
    real_t pix_area = 4.0 * PI / (12.0 * nside * nside);

    // Scale alm by pixel area (simple kernel)
    // For now, do on host
    complex_t* h_alm = new complex_t[n_maps * lp1 * lp1];
    CUDA_CHECK(cudaMemcpy(h_alm, alm_out, alm_size, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < (size_t)n_maps * lp1 * lp1; i++) {
        h_alm[i].x *= pix_area;
        h_alm[i].y *= pix_area;
    }

    CUDA_CHECK(cudaMemcpy(alm_out, h_alm, alm_size, cudaMemcpyHostToDevice));
    delete[] h_alm;

    // Cleanup
    CUDA_CHECK(cudaFree(phi_buffer));
    CUDA_CHECK(cudaFree(Gmy_buffer));
    CUDA_CHECK(cudaFree(ylm_buffer));
    CUDA_CHECK(cudaFree(ylm_south_buffer));
    CUDA_CHECK(cudaFree(map_ring_buffer));
    free_ring_geometry(geom);
}
