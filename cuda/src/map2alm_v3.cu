/**
 * map2alm_v3: Fused Per-Ring Parallel Implementation
 *
 * Algorithm:
 *   For each ring pair (north + south) in parallel:
 *     1. Compute cos(theta) on-the-fly
 *     2. FFT both rings -> Gm_north[m], Gm_south[m]
 *     3. Ylm recurrence for this theta
 *     4. Accumulate: alm[l,m] += Ylm * (Gm_north + (-1)^(l+m) * Gm_south)
 *
 * Memory: O(l_max) per thread block (no global Ylm storage)
 * Parallelism: n_north_rings independent thread blocks
 */

#include <cufft.h>
#include "../include/spht_types.h"
#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ============================================================================
// Device helper functions
// ============================================================================

__device__ __forceinline__
void compute_ring_geometry(int ring_idx, int nside,
                           double* cos_theta, double* sin_theta,
                           double* phi_0, int* n_pixels) {
    int ring_i = ring_idx + 1;  // 1-indexed

    if (ring_i < nside) {
        // North polar cap
        double i2_over_3n2 = (double)(ring_i * ring_i) / (3.0 * nside * nside);
        *cos_theta = 1.0 - i2_over_3n2;
        *sin_theta = sqrt(1.0 - (*cos_theta) * (*cos_theta));
        *phi_0 = M_PI / (2.0 * ring_i) * 0.5;
        *n_pixels = 4 * ring_i;
    } else if (ring_i > 3 * nside) {
        // South polar cap
        int mirror_i = 4 * nside - ring_i;
        double i2_over_3n2 = (double)(mirror_i * mirror_i) / (3.0 * nside * nside);
        *cos_theta = -(1.0 - i2_over_3n2);
        *sin_theta = sqrt(1.0 - (*cos_theta) * (*cos_theta));
        *phi_0 = M_PI / (2.0 * mirror_i) * 0.5;
        *n_pixels = 4 * mirror_i;
    } else {
        // Equatorial belt
        *cos_theta = 4.0 / 3.0 - 2.0 * ring_i / (3.0 * nside);
        *sin_theta = sqrt(1.0 - (*cos_theta) * (*cos_theta));
        int s = (ring_i % 2 == 0) ? 1 : 2;
        *phi_0 = M_PI / (2.0 * nside) * (1.0 - s / 2.0);
        *n_pixels = 4 * nside;
    }
}

// ============================================================================
// Simple DFT for small rings (used in shared memory)
// ============================================================================

__device__ void compute_dft_shared(
    const double* ring_data,  // Input: n_pixels values
    int n_pixels,
    int l_max,
    double phi_0,
    double2* Gm              // Output: l_max+1 complex values
) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Each thread computes some m values
    for (int m = tid; m <= l_max; m += block_size) {
        double re = 0.0, im = 0.0;

        // Handle aliasing for polar rings: m_eff = m % n_pixels
        int m_eff = m % n_pixels;

        for (int j = 0; j < n_pixels; j++) {
            double phi_j = 2.0 * M_PI * m_eff * j / n_pixels;
            re += ring_data[j] * cos(phi_j);
            im -= ring_data[j] * sin(phi_j);
        }

        // Apply phase correction for phi_0
        double phase = -m * phi_0;
        double cos_phase = cos(phase);
        double sin_phase = sin(phase);

        Gm[m].x = re * cos_phase - im * sin_phase;
        Gm[m].y = re * sin_phase + im * cos_phase;
    }

    __syncthreads();
}

// ============================================================================
// Compute Ylm column and accumulate to alm in one pass
// This avoids storing the full column
// ============================================================================

__device__ void ylm_accumulate_column(
    int m,
    int l_max,
    double cos_theta,
    double sin_theta,
    double2 gm_north,
    double2 gm_south,
    bool is_equator,
    int lp1,
    double* alm_real,
    double* alm_imag
) {
    // Y[m,m] = (-1)^m * sin^m(theta) * sqrt((2m+1)!!/(2m)!!) / sqrt(4*pi)
    double Ymm = 1.0 / sqrt(4.0 * M_PI);  // Y[0,0]

    for (int k = 1; k <= m; k++) {
        Ymm *= -sin_theta * sqrt((2.0 * k + 1.0) / (2.0 * k));
    }

    // Accumulate Y[m,m]
    {
        int l = m;
        int sign_ns = ((l + m) % 2 == 0) ? 1 : -1;
        double gm_re = gm_north.x + (is_equator ? 0.0 : sign_ns * gm_south.x);
        double gm_im = gm_north.y + (is_equator ? 0.0 : sign_ns * gm_south.y);
        atomicAdd(&alm_real[l * lp1 + m], Ymm * gm_re);
        atomicAdd(&alm_imag[l * lp1 + m], Ymm * gm_im);
    }

    if (m == l_max) return;

    // Y[m+1,m] = cos(theta) * sqrt(2m+3) * Y[m,m]
    double Ym1m = cos_theta * sqrt(2.0 * m + 3.0) * Ymm;

    // Accumulate Y[m+1,m]
    {
        int l = m + 1;
        int sign_ns = ((l + m) % 2 == 0) ? 1 : -1;
        double gm_re = gm_north.x + (is_equator ? 0.0 : sign_ns * gm_south.x);
        double gm_im = gm_north.y + (is_equator ? 0.0 : sign_ns * gm_south.y);
        atomicAdd(&alm_real[l * lp1 + m], Ym1m * gm_re);
        atomicAdd(&alm_imag[l * lp1 + m], Ym1m * gm_im);
    }

    if (m + 1 == l_max) return;

    // Recurrence: Y[l,m] = A[l,m] * cos(theta) * Y[l-1,m] - B[l,m] * Y[l-2,m]
    double Ylm2 = Ymm;   // Y[l-2,m]
    double Ylm1 = Ym1m;  // Y[l-1,m]

    for (int l = m + 2; l <= l_max; l++) {
        double l2 = (double)(l * l);
        double m2 = (double)(m * m);
        double lm1_2 = (double)((l - 1) * (l - 1));

        double A_lm = sqrt((4.0 * l2 - 1.0) / (l2 - m2));
        double B_lm = sqrt((2.0 * l + 1.0) / (2.0 * l - 3.0) * (lm1_2 - m2) / (l2 - m2));

        double Ylm = A_lm * cos_theta * Ylm1 - B_lm * Ylm2;

        // Accumulate Y[l,m]
        int sign_ns = ((l + m) % 2 == 0) ? 1 : -1;
        double gm_re = gm_north.x + (is_equator ? 0.0 : sign_ns * gm_south.x);
        double gm_im = gm_north.y + (is_equator ? 0.0 : sign_ns * gm_south.y);
        atomicAdd(&alm_real[l * lp1 + m], Ylm * gm_re);
        atomicAdd(&alm_imag[l * lp1 + m], Ylm * gm_im);

        Ylm2 = Ylm1;
        Ylm1 = Ylm;
    }
}

// ============================================================================
// Main fused kernel: one thread block per ring pair
// ============================================================================

__global__ void map2alm_fused_kernel(
    int nside,
    int l_max,
    int n_maps,
    int n_rings,
    const double* __restrict__ map_in,   // [n_maps, n_rings, max_ring_pixels]
    double* __restrict__ alm_real,       // [n_maps, lp1, lp1] - real parts
    double* __restrict__ alm_imag        // [n_maps, lp1, lp1] - imag parts
) {
    // Block index = north ring index (0 to n_north_rings-1)
    int north_ring_idx = blockIdx.x;
    int n_north_rings = 2 * nside;  // Including equator
    int lp1 = l_max + 1;
    int max_ring_pixels = 4 * nside;

    if (north_ring_idx >= n_north_rings) return;

    int south_ring_idx = n_rings - 1 - north_ring_idx;
    bool is_equator = (north_ring_idx == 2 * nside - 1);

    // Shared memory layout:
    // - ring_north[max_ring_pixels]: north ring data
    // - ring_south[max_ring_pixels]: south ring data
    // - Gm_north[lp1]: FFT of north ring
    // - Gm_south[lp1]: FFT of south ring
    extern __shared__ char shared_mem[];

    double* ring_north = (double*)shared_mem;
    double* ring_south = ring_north + max_ring_pixels;
    double2* Gm_north = (double2*)(ring_south + max_ring_pixels);
    double2* Gm_south = Gm_north + lp1;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // 1. Compute ring geometry
    double cos_theta_n, sin_theta_n, phi_0_n;
    int n_pixels_n;
    compute_ring_geometry(north_ring_idx, nside, &cos_theta_n, &sin_theta_n, &phi_0_n, &n_pixels_n);

    double phi_0_s = 0.0;
    int n_pixels_s = 0;
    if (!is_equator) {
        double cos_theta_s, sin_theta_s;
        compute_ring_geometry(south_ring_idx, nside, &cos_theta_s, &sin_theta_s, &phi_0_s, &n_pixels_s);
    }

    // Process each map
    for (int t = 0; t < n_maps; t++) {
        // 2. Load ring data to shared memory
        const double* map_t = map_in + (size_t)t * n_rings * max_ring_pixels;

        for (int j = tid; j < n_pixels_n; j += block_size) {
            ring_north[j] = map_t[north_ring_idx * max_ring_pixels + j];
        }
        if (!is_equator) {
            for (int j = tid; j < n_pixels_s; j += block_size) {
                ring_south[j] = map_t[south_ring_idx * max_ring_pixels + j];
            }
        }
        __syncthreads();

        // 3. Compute DFT -> Gm
        compute_dft_shared(ring_north, n_pixels_n, l_max, phi_0_n, Gm_north);
        if (!is_equator) {
            compute_dft_shared(ring_south, n_pixels_s, l_max, phi_0_s, Gm_south);
        } else {
            // Zero out south Gm for equator
            for (int m = tid; m <= l_max; m += block_size) {
                Gm_south[m] = make_double2(0.0, 0.0);
            }
        }
        __syncthreads();

        // 4. For each m, compute Ylm column and accumulate to alm
        double* alm_real_t = alm_real + (size_t)t * lp1 * lp1;
        double* alm_imag_t = alm_imag + (size_t)t * lp1 * lp1;

        for (int m = tid; m <= l_max; m += block_size) {
            double2 gm_n = Gm_north[m];
            double2 gm_s = Gm_south[m];

            ylm_accumulate_column(m, l_max, cos_theta_n, sin_theta_n,
                                  gm_n, gm_s, is_equator, lp1,
                                  alm_real_t, alm_imag_t);
        }
        __syncthreads();
    }
}

// ============================================================================
// Kernel to apply pixel area normalization and combine real/imag
// ============================================================================

__global__ void finalize_alm_kernel(
    int n_maps,
    int lp1,
    double pix_area,
    const double* __restrict__ alm_real,
    const double* __restrict__ alm_imag,
    complex_t* __restrict__ alm_out
) {
    int t = blockIdx.x;
    int l = blockIdx.y * blockDim.x + threadIdx.x;
    int m = blockIdx.z * blockDim.y + threadIdx.y;

    if (t >= n_maps || l >= lp1 || m > l) return;

    size_t idx = (size_t)t * lp1 * lp1 + l * lp1 + m;
    alm_out[idx].x = alm_real[idx] * pix_area;
    alm_out[idx].y = alm_imag[idx] * pix_area;
}

// ============================================================================
// Main entry point
// ============================================================================

extern "C"
void map2alm_cuda_v3(int nside, int l_max, int n_maps,
                     const real_t* map_in, complex_t* alm_out) {
    int n_rings = 4 * nside - 1;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;
    int max_ring_pixels = 4 * nside;

    // Allocate separate real/imag buffers for atomic accumulation
    double *alm_real, *alm_imag;
    size_t alm_size = (size_t)n_maps * lp1 * lp1 * sizeof(double);
    CUDA_CHECK(cudaMalloc(&alm_real, alm_size));
    CUDA_CHECK(cudaMalloc(&alm_imag, alm_size));
    CUDA_CHECK(cudaMemset(alm_real, 0, alm_size));
    CUDA_CHECK(cudaMemset(alm_imag, 0, alm_size));

    // Shared memory size per block
    size_t shared_size = 2 * max_ring_pixels * sizeof(double)   // ring_north + ring_south
                       + 2 * lp1 * sizeof(double2);             // Gm_north + Gm_south

    // Check shared memory limit
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (shared_size > prop.sharedMemPerBlock) {
        fprintf(stderr, "Error: Required shared memory %zu exceeds limit %zu\n",
                shared_size, prop.sharedMemPerBlock);
        cudaFree(alm_real);
        cudaFree(alm_imag);
        return;
    }

    // Launch fused kernel: one block per north ring
    int block_size = 256;  // Threads per block
    int grid_size = n_north_rings;

    map2alm_fused_kernel<<<grid_size, block_size, shared_size>>>(
        nside, l_max, n_maps, n_rings, map_in, alm_real, alm_imag
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Finalize: combine real/imag and apply pixel area
    double pix_area = 4.0 * M_PI / (12.0 * nside * nside);

    dim3 block_final(16, 16);
    dim3 grid_final(n_maps, CEILDIV(lp1, 16), CEILDIV(lp1, 16));
    finalize_alm_kernel<<<grid_final, block_final>>>(
        n_maps, lp1, pix_area, alm_real, alm_imag, alm_out
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    cudaFree(alm_real);
    cudaFree(alm_imag);
}
