/**
 * map2alm_v4: Batched Per-Ring Parallel Implementation
 *
 * Same algorithm as v3 but with pixel batching to stay within shared memory limits.
 * Pixels are loaded in batches, with Gm accumulated in thread-local storage.
 */

#include <cufft.h>
#include "../include/spht_types.h"
#include <stdio.h>

// Batch size for loading ring pixels - tune to stay under shared memory limit
// 2 * PIXEL_BATCH_SIZE * sizeof(double) should be < 48KB
#define PIXEL_BATCH_SIZE 2048

// Max m values per thread (for thread-local Gm storage)
// Should be >= ceil((3*nside) / block_size) for largest expected nside
#define MAX_M_PER_THREAD 64

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
// Compute Ylm column and accumulate to alm in one pass
// ============================================================================

__device__ void ylm_accumulate_column_v4(
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
// Main fused kernel with pixel batching
// ============================================================================

__global__ void map2alm_fused_kernel_v4(
    int nside,
    int l_max,
    int n_maps,
    int n_rings,
    const double* __restrict__ map_in,
    double* __restrict__ alm_real,
    double* __restrict__ alm_imag
) {
    int north_ring_idx = blockIdx.x;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;
    int max_ring_pixels = 4 * nside;

    if (north_ring_idx >= n_north_rings) return;

    int south_ring_idx = n_rings - 1 - north_ring_idx;
    bool is_equator = (north_ring_idx == 2 * nside - 1);

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Shared memory for pixel batches only
    extern __shared__ double shared_mem[];
    double* ring_north_batch = shared_mem;
    double* ring_south_batch = ring_north_batch + PIXEL_BATCH_SIZE;

    // Compute ring geometry
    double cos_theta_n, sin_theta_n, phi_0_n;
    int n_pixels_n;
    compute_ring_geometry(north_ring_idx, nside, &cos_theta_n, &sin_theta_n, &phi_0_n, &n_pixels_n);

    double cos_theta_s, sin_theta_s, phi_0_s;
    int n_pixels_s = 0;
    if (!is_equator) {
        compute_ring_geometry(south_ring_idx, nside, &cos_theta_s, &sin_theta_s, &phi_0_s, &n_pixels_s);
    }

    // Thread-local storage for Gm values
    // Each thread handles m = tid, tid+block_size, tid+2*block_size, ...
    double gm_north_re[MAX_M_PER_THREAD];
    double gm_north_im[MAX_M_PER_THREAD];
    double gm_south_re[MAX_M_PER_THREAD];
    double gm_south_im[MAX_M_PER_THREAD];

    // Process each map
    for (int t = 0; t < n_maps; t++) {
        const double* map_t = map_in + (size_t)t * n_rings * max_ring_pixels;

        // Initialize thread-local Gm to zero
        for (int i = 0; i < MAX_M_PER_THREAD; i++) {
            gm_north_re[i] = 0.0;
            gm_north_im[i] = 0.0;
            gm_south_re[i] = 0.0;
            gm_south_im[i] = 0.0;
        }

        // Process north ring pixels in batches
        for (int pix_start = 0; pix_start < n_pixels_n; pix_start += PIXEL_BATCH_SIZE) {
            int batch_size = min(PIXEL_BATCH_SIZE, n_pixels_n - pix_start);

            // Cooperative load of pixel batch
            for (int j = tid; j < batch_size; j += block_size) {
                ring_north_batch[j] = map_t[north_ring_idx * max_ring_pixels + pix_start + j];
            }
            __syncthreads();

            // Each thread accumulates DFT for its m values
            int m_idx = 0;
            for (int m = tid; m <= l_max; m += block_size, m_idx++) {
                int m_eff = m % n_pixels_n;
                for (int jj = 0; jj < batch_size; jj++) {
                    int j = pix_start + jj;
                    double phi_j = 2.0 * M_PI * m_eff * j / n_pixels_n;
                    double c, s;
                    sincos(phi_j, &s, &c);
                    gm_north_re[m_idx] += ring_north_batch[jj] * c;
                    gm_north_im[m_idx] -= ring_north_batch[jj] * s;
                }
            }
            __syncthreads();
        }

        // Process south ring pixels in batches (if not equator)
        if (!is_equator) {
            for (int pix_start = 0; pix_start < n_pixels_s; pix_start += PIXEL_BATCH_SIZE) {
                int batch_size = min(PIXEL_BATCH_SIZE, n_pixels_s - pix_start);

                // Cooperative load of pixel batch
                for (int j = tid; j < batch_size; j += block_size) {
                    ring_south_batch[j] = map_t[south_ring_idx * max_ring_pixels + pix_start + j];
                }
                __syncthreads();

                // Each thread accumulates DFT for its m values
                int m_idx = 0;
                for (int m = tid; m <= l_max; m += block_size, m_idx++) {
                    int m_eff = m % n_pixels_s;
                    for (int jj = 0; jj < batch_size; jj++) {
                        int j = pix_start + jj;
                        double phi_j = 2.0 * M_PI * m_eff * j / n_pixels_s;
                        double c, s;
                        sincos(phi_j, &s, &c);
                        gm_south_re[m_idx] += ring_south_batch[jj] * c;
                        gm_south_im[m_idx] -= ring_south_batch[jj] * s;
                    }
                }
                __syncthreads();
            }
        }

        // Apply phase corrections and do Ylm accumulation
        double* alm_real_t = alm_real + (size_t)t * lp1 * lp1;
        double* alm_imag_t = alm_imag + (size_t)t * lp1 * lp1;

        int m_idx = 0;
        for (int m = tid; m <= l_max; m += block_size, m_idx++) {
            // Apply phase correction for north ring
            double phase_n = -m * phi_0_n;
            double cos_pn, sin_pn;
            sincos(phase_n, &sin_pn, &cos_pn);
            double2 gm_n;
            gm_n.x = gm_north_re[m_idx] * cos_pn - gm_north_im[m_idx] * sin_pn;
            gm_n.y = gm_north_re[m_idx] * sin_pn + gm_north_im[m_idx] * cos_pn;

            // Apply phase correction for south ring
            double2 gm_s = make_double2(0.0, 0.0);
            if (!is_equator) {
                double phase_s = -m * phi_0_s;
                double cos_ps, sin_ps;
                sincos(phase_s, &sin_ps, &cos_ps);
                gm_s.x = gm_south_re[m_idx] * cos_ps - gm_south_im[m_idx] * sin_ps;
                gm_s.y = gm_south_re[m_idx] * sin_ps + gm_south_im[m_idx] * cos_ps;
            }

            // Do Ylm recurrence and accumulate to alm
            ylm_accumulate_column_v4(m, l_max, cos_theta_n, sin_theta_n,
                                  gm_n, gm_s, is_equator, lp1,
                                  alm_real_t, alm_imag_t);
        }
        __syncthreads();
    }
}

// ============================================================================
// Finalization kernel
// ============================================================================

__global__ void finalize_alm_kernel_v4(
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
void map2alm_cuda_v4(int nside, int l_max, int n_maps,
                     const real_t* map_in, complex_t* alm_out) {
    int n_rings = 4 * nside - 1;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;

    // Check that l_max doesn't exceed our per-thread storage
    int block_size = 256;
    int m_per_thread = (lp1 + block_size - 1) / block_size;
    if (m_per_thread > MAX_M_PER_THREAD) {
        fprintf(stderr, "Error: l_max=%d requires %d m values per thread, max is %d\n",
                l_max, m_per_thread, MAX_M_PER_THREAD);
        fprintf(stderr, "Increase MAX_M_PER_THREAD or block_size\n");
        return;
    }

    // Allocate separate real/imag buffers for atomic accumulation
    double *alm_real, *alm_imag;
    size_t alm_size = (size_t)n_maps * lp1 * lp1 * sizeof(double);
    CUDA_CHECK(cudaMalloc(&alm_real, alm_size));
    CUDA_CHECK(cudaMalloc(&alm_imag, alm_size));
    CUDA_CHECK(cudaMemset(alm_real, 0, alm_size));
    CUDA_CHECK(cudaMemset(alm_imag, 0, alm_size));

    // Shared memory: just for pixel batches
    size_t shared_size = 2 * PIXEL_BATCH_SIZE * sizeof(double);

    // Verify shared memory fits
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

    // Launch fused kernel
    int grid_size = n_north_rings;
    map2alm_fused_kernel_v4<<<grid_size, block_size, shared_size>>>(
        nside, l_max, n_maps, n_rings, map_in, alm_real, alm_imag
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Finalize: combine real/imag and apply pixel area
    double pix_area = 4.0 * M_PI / (12.0 * nside * nside);

    dim3 block_final(16, 16);
    dim3 grid_final(n_maps, CEILDIV(lp1, 16), CEILDIV(lp1, 16));
    finalize_alm_kernel_v4<<<grid_final, block_final>>>(
        n_maps, lp1, pix_area, alm_real, alm_imag, alm_out
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    cudaFree(alm_real);
    cudaFree(alm_imag);
}
